/******************************************************************************
 * camera_stream.c
 *
 * Continuous camera-driven inference with **live image streaming**.
 * Identical to camera_inference.c (capture → preprocess → CNN → logits)
 * but additionally streams the 32x32 RGB888 image that was fed to the
 * CNN, so the host can render a livestream alongside the probability
 * distribution.
 *
 * Companion host:    host/host_camera_stream.py
 * Build / flash:     bash_device_scripts/device_camera_stream.sh <variant>
 * Live viewer:       bash_device_scripts/host_camera_stream.sh   <variant>
 *
 * Wire protocol (one packet per inference, all little-endian):
 *
 *     dev -> host:  uint8   magic[4]      = 0xA5 0xA5 0xCD 0xCD
 *                   uint32  frame_counter
 *                   uint32  cnn_cycles
 *                   int32   logits[10]
 *                   uint8   img_rgb888[32*32*3]   // 3072 bytes
 *     total = 4 + 4 + 4 + 40 + 3072 = 3124 bytes / frame
 *
 * Why a 4-byte magic instead of camera_inference.c's 1-byte sync?
 *   Image bytes can take any value in [0, 255], so a single sync byte
 *   would collide with pixel data. A 4-byte fixed magic is collision-
 *   resistant enough for raw camera content.
 *
 * UART throughput note:
 *   Runs at 115200 baud (matches camera_inference.c) — the 3124-byte
 *   packet takes ~270 ms, giving ~3.7 FPS for the livestream.
 *   Higher baud rates (921600+) would lift FPS to ~25 but the MAX78000
 *   PCLK at 50 MHz can't synthesize them cleanly from integer dividers,
 *   so MXC_UART_SetFrequency silently fails and the host ends up reading
 *   garbage at the wrong baud. Sticking to 115200 keeps the pipeline
 *   robust at the cost of FPS.
 *
 * Same image normalization as inference_test_set.c / camera_inference.c:
 *     RGB565 -> RGB888 -> centered int8  = pixel - 128
 *     packed as (B<<16) | (G<<8) | R, one uint32 per pixel, into CNN SRAM.
 * Display image is the SAME RGB888 frame BEFORE the int8 centering — the
 * uncentered uint8 pixels so the host sees natural colours, not a
 * grey-shifted version.
 *****************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "board.h"
#include "led.h"
#include "uart.h"
#include "camera.h"
#include "dma.h"

#include "cnn.h"

/* ---- camera / CNN-input geometry ----------------------------------------- */

#define IMAGE_SIZE_X   64
#define IMAGE_SIZE_Y   64
#define CNN_INPUT_W    32
#define CNN_INPUT_H    32
#define CAMERA_FREQ    10000000  /* 10 MHz */

#define UART_BAUD      115200    /* keep equal to Board_Init() default —
                                  * PCLK can't cleanly divide higher rates */

/* ---- UART ---------------------------------------------------------------- */

#define UART_REGS  MXC_UART_GET_UART(CONSOLE_UART)

/* 4-byte sync — see header. Image bytes can be any value so a 1-byte
 * sync would mis-trigger. */
static const uint8_t MAGIC[4] = { 0xA5, 0xA5, 0xCD, 0xCD };

/* CNN ISR (in cnn.c) writes the accelerator's cycle count here. */
volatile uint32_t cnn_time;

static int dma_channel = -1;

/* RGB888 display copy — the uncentered, downsampled frame the CNN saw.
 * 32*32*3 = 3072 bytes. Lives in regular SRAM (not CNN SRAM). */
static uint8_t display_rgb888[CNN_INPUT_W * CNN_INPUT_H * 3];

/* ---- UART byte I/O (blocking) ------------------------------------------- */

static void uart_write_byte(uint8_t b)
{
    MXC_UART_WriteCharacter(UART_REGS, b);
}

static void uart_write_bytes(const uint8_t *buf, int n)
{
    for (int i = 0; i < n; ++i) uart_write_byte(buf[i]);
}

/* ---- Frame preprocessing ------------------------------------------------- */

/* Same downsample as camera_inference.c, BUT also writes the uncentered
 * RGB888 pixel into `display_rgb888` for streaming to the host. */
static void load_camera_frame(const uint8_t *raw_rgb565)
{
    uint32_t *cnn_input = (uint32_t *)0x50400000;

    for (int y = 0; y < CNN_INPUT_H; ++y) {
        for (int x = 0; x < CNN_INPUT_W; ++x) {
            int src_y = y * 2;
            int src_x = x * 2;
            int src_idx = (src_y * IMAGE_SIZE_X + src_x) * 2;

            uint16_t pixel = ((uint16_t)raw_rgb565[src_idx] << 8)
                           |  (uint16_t)raw_rgb565[src_idx + 1];

            uint8_t r5 = (pixel >> 11) & 0x1F;
            uint8_t g6 = (pixel >>  5) & 0x3F;
            uint8_t b5 =  pixel        & 0x1F;
            uint8_t r = (r5 << 3) | (r5 >> 2);
            uint8_t g = (g6 << 2) | (g6 >> 4);
            uint8_t b = (b5 << 3) | (b5 >> 2);

            /* (1) Display copy — natural uint8 RGB888 for the host */
            int disp_idx = (y * CNN_INPUT_W + x) * 3;
            display_rgb888[disp_idx + 0] = r;
            display_rgb888[disp_idx + 1] = g;
            display_rgb888[disp_idx + 2] = b;

            /* (2) CNN copy — ai8x int8 centering, packed BGR-style */
            uint8_t r_i8 = (uint8_t)((int)r - 128);
            uint8_t g_i8 = (uint8_t)((int)g - 128);
            uint8_t b_i8 = (uint8_t)((int)b - 128);

            uint32_t packed = ((uint32_t)b_i8 << 16)
                            | ((uint32_t)g_i8 <<  8)
                            |  (uint32_t)r_i8;

            cnn_input[y * CNN_INPUT_W + x] = packed;
        }
    }
}

/* ---- main --------------------------------------------------------------- */

int main(void)
{
    int ret;

    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Board_Init brings the console UART up at 115200 — same rate we
     * stream packets at, so no reconfiguration needed. (Tried 921600
     * once: MXC_UART_SetFrequency returns OK but PCLK divides poorly
     * and the resulting baud is wrong; the host then reads garbage.
     * Sticking to 115200 keeps things simple and matches
     * camera_inference.c, at the cost of ~3.7 FPS instead of ~25.) */
    Board_Init();

    printf("\r\nBOOT camera_stream @%d baud\r\n", UART_BAUD);
    printf("BOOT step 1/4: Board_Init ok\r\n");

    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_GPIO0);
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_GPIO1);

    ret = MXC_DMA_Init();
    if (ret != E_NO_ERROR) {
        printf("ERROR: MXC_DMA_Init failed (%d)\r\n", ret);
        while (1) { __WFI(); }
    }
    dma_channel = MXC_DMA_AcquireChannel();
    if (dma_channel < 0) {
        printf("ERROR: MXC_DMA_AcquireChannel failed (%d)\r\n", dma_channel);
        while (1) { __WFI(); }
    }
    printf("BOOT step 2/4: DMA init + channel %d acquired\r\n", dma_channel);

    ret = camera_init(CAMERA_FREQ);
    if (ret != E_NO_ERROR) {
        printf("ERROR: camera_init failed (%d)\r\n", ret);
        while (1) { __WFI(); }
    }
    printf("BOOT step 3/4: camera_init ok @ %d Hz\r\n", CAMERA_FREQ);

    ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y,
                       PIXFORMAT_RGB565,
                       FIFO_FOUR_BYTE, USE_DMA, dma_channel);
    if (ret != E_NO_ERROR) {
        printf("ERROR: camera_setup failed (%d)\r\n", ret);
        while (1) { __WFI(); }
    }
    printf("BOOT step 4/4: camera ok (%dx%d RGB565 -> 32x32)\r\n",
           IMAGE_SIZE_X, IMAGE_SIZE_Y);

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK,
               MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    printf("BOOT cnn ok — streaming begins "
           "(4-byte magic 0xA5 0xA5 0xCD 0xCD)\r\n");

    int32_t  logits[10];
    uint32_t frame_counter = 0;

    LED_On(0);

    for (;;) {
        /* 1. Capture */
        camera_start_capture_image();
        while (!camera_is_image_rcv()) { /* spin */ }

        uint8_t  *raw;
        uint32_t  imglen, w, h;
        camera_get_image(&raw, &imglen, &w, &h);

        /* 2. Preprocess: fill both CNN SRAM AND display_rgb888 */
        load_camera_frame(raw);

        /* 3. Inference */
        cnn_init();
        cnn_configure();
        cnn_time = 0;
        cnn_start();
        while (cnn_time == 0) { __WFI(); }

        /* 4. Read 10 int32 class logits */
        cnn_unload((uint32_t *)logits);

        /* 5. Stream one packet (3124 bytes total) */
        uart_write_bytes(MAGIC, sizeof(MAGIC));
        uart_write_bytes((const uint8_t *)&frame_counter, sizeof(frame_counter));
        uart_write_bytes((const uint8_t *)&cnn_time,      sizeof(cnn_time));
        uart_write_bytes((const uint8_t *)logits,         sizeof(logits));
        uart_write_bytes(display_rgb888,                  sizeof(display_rgb888));

        frame_counter++;
        LED_Toggle(0);
    }

    /* unreachable */
}

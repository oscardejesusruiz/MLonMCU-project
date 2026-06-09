/******************************************************************************
 * profile_camera.c
 *
 * Full-pipeline profiling firmware for the camera-driven inference path.
 *
 * Per inference, the firmware times TWO independent intervals:
 *
 *   1. End-to-end latency (t_e2e) — measured with the Cortex-M4 DWT cycle
 *      counter (CPU clock = 100 MHz). Brackets the WHOLE pipeline:
 *
 *          ┌─ camera_start_capture_image()
 *          │  spin on camera_is_image_rcv()
 *          │  camera_get_image()
 *          │  load_camera_frame()           (RGB565 → packed int8 SRAM)
 *          │  cnn_init()
 *          │  cnn_configure()
 *          │  cnn_start()
 *          │  wait for completion ISR (WFI)
 *          └─ cnn_unload()
 *
 *      UART transmission of the result packet is NOT included — it is
 *      reporting overhead, not part of the inference pipeline.
 *
 *   2. CNN latency (t_cnn) — what `MXC_TMR_SW_Stop` puts into `cnn_time`
 *      via the existing CNN completion ISR. The MSDK timer driver returns
 *      microseconds; that's what we forward to the host.
 *
 * Companion host:    host/host_profile.py
 * Build / flash:     bash_device_scripts/device_profile.sh <variant>
 * Live monitor:      bash_device_scripts/host_profile.sh   <variant>
 *
 * Wire protocol (one packet per inference, little-endian, 50 bytes total):
 *
 *     dev -> host:  uint8   sync           = 0xCB    (distinct from
 *                                                     camera_inference's 0xCC)
 *                   uint8   inference_id   (mod 256 — for drop detection)
 *                   uint32  e2e_us         (DWT-measured pipeline µs)
 *                   uint32  cnn_us         (MSDK-measured CNN µs)
 *                   int32   logits[10]     (sanity check — host can softmax)
 *
 * The host averages over a rolling N=10 window and prints:
 *
 *     t_e2e, t_cnn, CNN cycles (@50 MHz), MAC/cycle,
 *     E_e2e, E_cnn, P/MHz (CPU & CNN), TOPS/W (peak & measured)
 *
 * Same drop-in-replacement-for-main.c pattern as camera_inference.c.
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
#include "tmr.h"

#include "cnn.h"

/* End-to-end timer — uses MXC_TMR_SW_Start / MXC_TMR_SW_Stop, the same
 * pattern cnn.c uses for cnn_time on MXC_TMR0. Using TMR1 here so the
 * two measurements don't fight over the same peripheral. The MSDK timer
 * driver returns elapsed microseconds regardless of the underlying tick
 * rate — same units as cnn_time, no conversion needed.
 *
 * We tried the Cortex-M4 DWT cycle counter first; on the MAX78000 it
 * compiles fine but CYCCNT never increments (DWT is gated / disabled in
 * silicon). MSDK timers are the canonical Maxim path for cycle-accurate
 * measurement on this part. */
#define E2E_TIMER  MXC_TMR1

/* ---- camera / CNN-input geometry ----------------------------------------- */

#define IMAGE_SIZE_X   64
#define IMAGE_SIZE_Y   64
#define CNN_INPUT_W    32
#define CNN_INPUT_H    32
#define CAMERA_FREQ    18000000  /* 10 MHz */

/* ---- UART ---------------------------------------------------------------- */

#define UART_REGS  MXC_UART_GET_UART(CONSOLE_UART)
#define SYNC_BYTE  0xCB         /* different from camera_inference's 0xCC */

/* CNN ISR (in cnn.c) writes the accelerator's MSDK timer value here. */
volatile uint32_t cnn_time;

static int dma_channel = -1;

/* ---- UART byte I/O (blocking) ------------------------------------------- */

static void uart_write_byte(uint8_t b)
{
    MXC_UART_WriteCharacter(UART_REGS, b);
}

static void uart_write_bytes(const uint8_t *buf, int n)
{
    for (int i = 0; i < n; ++i) uart_write_byte(buf[i]);
}

/* ---- Frame preprocessing (same as camera_inference.c) ------------------- */

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

    Board_Init();

    printf("\r\nBOOT profile_camera @115200\r\n");
    printf("BOOT clocks: CPU=%lu Hz, CNN=PCLK/1=50 MHz\r\n",
           (unsigned long)SystemCoreClock);
    printf("BOOT step 1/4: Board_Init ok (e2e timer = MXC_TMR1)\r\n");

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
    printf("BOOT cnn ok — profiling stream begins (sync byte 0x%02X)\r\n",
           SYNC_BYTE);

    int32_t   logits[10];
    uint8_t   inference_id = 0;

    LED_On(0);

    for (;;) {
        /* === Start of measured pipeline ============================== */
        MXC_TMR_SW_Start(E2E_TIMER);

        /* 1. Camera capture (blocking — spin until DMA done) */
        camera_start_capture_image();
        while (!camera_is_image_rcv()) { /* spin */ }

        uint8_t  *raw;
        uint32_t  imglen, w, h;
        camera_get_image(&raw, &imglen, &w, &h);

        /* 2. Preprocess */
        load_camera_frame(raw);

        /* 3. CNN setup */
        cnn_init();
        cnn_configure();

        /* 4. CNN run — cnn_time is reset before cnn_start, then written by
         *    CNN_ISR via MXC_TMR_SW_Stop(TMR0). Units: µs. */
        cnn_time = 0;
        cnn_start();
        while (cnn_time == 0) { __WFI(); }

        /* 5. Read 10 int32 logits back from CNN data SRAM */
        cnn_unload((uint32_t *)logits);

        /* MXC_TMR_SW_Stop returns elapsed time in microseconds (MSDK
         * timer driver normalizes the unit regardless of tick rate).
         * Same convention as cnn_time. */
        uint32_t e2e_us = MXC_TMR_SW_Stop(E2E_TIMER);
        /* === End of measured pipeline ================================ */

        uint32_t cnn_us = cnn_time;

        /* 6. Stream one packet (50 bytes) — NOT included in e2e */
        uart_write_byte(SYNC_BYTE);
        uart_write_byte(inference_id);
        uart_write_bytes((const uint8_t *)&e2e_us, sizeof(e2e_us));
        uart_write_bytes((const uint8_t *)&cnn_us, sizeof(cnn_us));
        uart_write_bytes((const uint8_t *)logits,  sizeof(logits));

        inference_id++;
        LED_Toggle(0);
    }

    /* unreachable */
}

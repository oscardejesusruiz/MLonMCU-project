/******************************************************************************
 * camera_inference.c
 *
 * Continuous camera-driven inference for any of the synthesized variants.
 * Captures frames from the on-board OV7692 (FTHR_RevA), downsamples to
 * 32x32, runs through the CNN accelerator, and streams the 10 int32
 * logits plus frame ID and CNN cycle count back to the host over UART.
 *
 * Companion host:    host/host_camera.py
 * Build / flash:     bash_device_scripts/device_camera.sh <variant>
 * Live monitor:      bash_device_scripts/host_camera.sh   <variant>
 *
 * Wire protocol (one packet per inference, all little-endian):
 *
 * dev -> host:  uint8   sync          = 0xCC
 * uint32  frame_counter
 * uint32  cnn_cycles
 * int32   logits[10]
 * total = 1 + 4 + 4 + 40 = 49 bytes / frame
 *
 * The loop runs as fast as camera + inference allow (typically 10-30 FPS
 * on FTHR_RevA). UART overhead at 115200 baud for the 49-byte packet is
 * ~4.5 ms — usually negligible vs camera frame time.
 *
 * Same image normalization as `host/host_test_set.py`:
 * RGB565 -> RGB888 -> centered int8  = pixel - 128
 * packed as (B<<16) | (G<<8) | R, one uint32 per pixel
 * written directly into CNN data SRAM at 0x50400000.
 *
 * Drop-in replacement for `main.c` in the ai8xize-output project — same
 * pattern as inference_test_set.c / measure_inference.c.
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

/* ---- UART ---------------------------------------------------------------- */

#define UART_REGS  MXC_UART_GET_UART(CONSOLE_UART)
#define SYNC_BYTE  0xCC

/* CNN ISR (in cnn.c) writes the accelerator's cycle count here. */
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

/* ---- Frame preprocessing ------------------------------------------------- */

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

    /* Standard 100 MHz IPO clock, instruction cache on. */
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    Board_Init();

    printf("\r\nBOOT camera_inference @115200\r\n");
    printf("BOOT step 1/4: Board_Init ok\r\n");

    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_GPIO0);
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_GPIO1);

    ret = MXC_DMA_Init();
    if (ret != E_NO_ERROR) {
        printf("ERROR: MXC_DMA_Init failed (%d)\r\n", ret);
        while (1) {}
    }
    dma_channel = MXC_DMA_AcquireChannel();
    if (dma_channel < 0) {
        printf("ERROR: MXC_DMA_AcquireChannel failed (%d)\r\n", dma_channel);
        while (1) {}
    }
    printf("BOOT step 2/4: DMA init + channel %d acquired\r\n", dma_channel);

    ret = camera_init(CAMERA_FREQ);
    if (ret != E_NO_ERROR) {
        printf("ERROR: camera_init failed (%d)\r\n", ret);
        while (1) {}
    }
    printf("BOOT step 3/4: camera_init ok @ %d Hz\r\n", CAMERA_FREQ);

    ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y,
                       PIXFORMAT_RGB565,
                       FIFO_FOUR_BYTE, USE_DMA, dma_channel);
    if (ret != E_NO_ERROR) {
        printf("ERROR: camera_setup failed (%d)\r\n", ret);
        while (1) {}
    }
    printf("BOOT step 4/4: camera ok (%dx%d RGB565 -> 32x32)\r\n",
           IMAGE_SIZE_X, IMAGE_SIZE_Y);

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK,
               MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    printf("BOOT cnn ok — streaming begins (sync byte 0x%02X)\r\n", SYNC_BYTE);

    int32_t  logits[10];
    uint32_t frame_counter = 0;

    /* MODIFICACIÓ CLAU: Inicialització permanent de SysTick com a base de temps global */
    SysTick->LOAD = 0x00FFFFFFUL; 
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;

    LED_On(0);

    for (;;) {
        /* 1. Capture one frame (blocking) */
        camera_start_capture_image();
        while (!camera_is_image_rcv()) { /* spin until DMA done */ }

        uint8_t  *raw;
        uint32_t  imglen, w, h;
        camera_get_image(&raw, &imglen, &w, &h);

        /* 2. Preprocess: RGB565 -> int8 packed -> CNN SRAM */
        load_camera_frame(raw);

        /* 3. Run one inference */
        cnn_init();          /* clear SM state from previous run */
        cnn_configure();
        
        // Reset del comptador SysTick i captura del cicle inicial
        SysTick->VAL = 0;
        uint32_t t_start = SysTick->VAL;
        
        cnn_time = 0;
        cnn_start();
        
        /* MODIFICACIÓ CLAU: Espera activa amb barrera de memòria per evitar que la CPU s'adormi */
        while (cnn_time == 0) { 
            __asm volatile("" : : : "memory"); 
        }

        // Captura del cicle de rellotge final de CPU immediatament
        uint32_t t_end = SysTick->VAL;

        /* 4. Read back the 10 int32 class logits */
        cnn_unload((uint32_t *)logits);

        // Calculem la mètrica real de cicles transcorreguts (SysTick és decreixent)
        uint32_t elapsed_cycles = 0;
        if (t_start >= t_end) {
            elapsed_cycles = t_start - t_end;
        } else {
            elapsed_cycles = (0x00FFFFFFUL - t_end) + t_start;
        }

        /* 5. Stream one packet (49 bytes)
         * Enviem elapsed_cycles a la posició destinada a cnn_cycles per garantir 
         * dades reals i invariants respecte als entorns d'estalvi de consum de la placa. */
        uart_write_byte(SYNC_BYTE);
        uart_write_bytes((const uint8_t *)&frame_counter, sizeof(frame_counter));
        uart_write_bytes((const uint8_t *)&elapsed_cycles, sizeof(elapsed_cycles));
        uart_write_bytes((const uint8_t *)logits,          sizeof(logits));

        frame_counter++;
        LED_Toggle(0);
    }
}
/******************************************************************************
 * inference_test_set.c
 *
 * Receive CIFAR-10 test images one at a time over UART, run inference on the
 * CNN accelerator, return the int32 logits + CNN cycle count to the host.
 *
 * Companion host script: ../host/host_test_set.py
 *
 * Wire protocol (matches host script):
 *
 * per image:
 * host -> dev:  1 byte sync 0xAA  +  3072 bytes int8 image (CHW, row-major)
 * dev  -> host: 1 byte sync 0xBB  +  uint32 cnn_cycles  +  10 * int32 logits
 *
 * Drop this file in place of the generated `main.c` in the ai8xize-output
 * project. The generated `cnn.c`, `cnn.h`, `weights.h` stay untouched.
 *
 * Build:
 * make BOARD=EvKit_V1
 * make TARGET=MAX78000 BOARD=EvKit_V1 flash
 *
 * Then on the host:
 * uv run python host/host_test_set.py --port /dev/cu.usbmodemXXXX --tag X
 *****************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "led.h"
#include "uart.h"
#include "board.h"

#include "cnn.h"

/* ------------- UART configuration ----------------------------------------- */

#define UART_REGS         MXC_UART_GET_UART(CONSOLE_UART)

#define SYNC_REQ          0xAA
#define SYNC_REP          0xBB
#define IMG_BYTES         (3 * 32 * 32)   /* 3072 */
#define N_CLASSES         10

/* ------------- CNN input plumbing ----------------------------------------- */

uint32_t input_0[1024];

/* Forward decl — defined in the synthesizer-generated cnn.c. */
extern void memcpy32(uint32_t *dst, const uint32_t *src, int n);

static void load_user_input(const int8_t *img_chw)
{
    /* img_chw layout: 3 planes of 32*32, plane order R, G, B. */
    const int8_t *r = img_chw + 0 * 1024;
    const int8_t *g = img_chw + 1 * 1024;
    const int8_t *b = img_chw + 2 * 1024;
    for (int i = 0; i < 1024; ++i) {
        uint32_t w = ((uint8_t)r[i])
                   | ((uint32_t)((uint8_t)g[i]) << 8)
                   | ((uint32_t)((uint8_t)b[i]) << 16);
        input_0[i] = w;
    }
    memcpy32((uint32_t *) 0x50400000, input_0, 1024);
}

/* ------------- UART byte I/O (blocking) ----------------------------------- */

static void uart_init_console(void)
{
    Board_Init();
}

static uint8_t uart_read_byte(void)
{
    int c;
    do {
        c = MXC_UART_ReadCharacterRaw(UART_REGS);
    } while (c < 0);
    return (uint8_t)c;
}

static void uart_read_bytes(uint8_t *buf, int n)
{
    for (int i = 0; i < n; ++i) buf[i] = uart_read_byte();
}

static void uart_write_byte(uint8_t b)
{
    MXC_UART_WriteCharacter(UART_REGS, b);
}

static void uart_write_bytes(const uint8_t *buf, int n)
{
    for (int i = 0; i < n; ++i) uart_write_byte(buf[i]);
}

/* ------------- main ------------------------------------------------------- */

volatile uint32_t cnn_time;

int main(void)
{
    /* Same clocking + CNN init as the generated main.c */
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    uart_init_console();
    printf("\r\nBOOT inference_test_set @115200\r\n");

    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_GPIO0);
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_GPIO1);
    printf("BOOT gpio ok\r\n");

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    printf("BOOT cnn ok, waiting sync 0xAA...\r\n");

    /* Reusable image buffer (CHW int8) */
    static int8_t img[IMG_BYTES];
    int32_t logits[N_CLASSES];

    /* Configurem el SysTick com a comptador de temps global per a la CPU */
    SysTick->LOAD = 0x00FFFFFFUL; 
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;

    LED_On(0);

    for (;;) {
        /* 1. wait for sync byte */
        uint8_t s;
        do { s = uart_read_byte(); } while (s != SYNC_REQ);

        /* 2. receive 3072 bytes int8 image */
        uart_read_bytes((uint8_t *)img, IMG_BYTES);

        /* 3. Full re-init + configure + load */
        cnn_init();
        cnn_configure();
        load_user_input(img);
        
        // Reset del comptador SysTick i captura inicial dels cicles de la CPU
        SysTick->VAL = 0;
        uint32_t t_start = SysTick->VAL;
        
        cnn_time = 0;
        cnn_start();
        
        /* MODIFICACIÓ CLAU: Espera activa segura amb barrera de memòria. 
         * Evita de manera radical l'ús de __WFI() que desactiva busos de temps. */
        while (cnn_time == 0) { 
            __asm volatile("" : : : "memory"); 
        }

        // Captura del cicle final de CPU immediatament
        uint32_t t_end = SysTick->VAL;

        /* 4. read 10 int32 logits from output region */
        cnn_unload((uint32_t *)logits);

        // Calculem la mètrica real de cicles transcorreguts (SysTick decreix)
        uint32_t elapsed_cycles = 0;
        if (t_start >= t_end) {
            elapsed_cycles = t_start - t_end;
        } else {
            elapsed_cycles = (0x00FFFFFFUL - t_end) + t_start;
        }

        /* 5. respond: 0xBB + elapsed_cycles (LE u32) + 10 logits (LE i32)
         * Enviem elapsed_cycles en comptes de cnn_time per garantir que el script
         * del host rep cicles de rellotge reals enters no nuls. */
        uart_write_byte(SYNC_REP);
        uart_write_bytes((const uint8_t *)&elapsed_cycles, sizeof(elapsed_cycles));
        uart_write_bytes((const uint8_t *)logits, sizeof(logits));

        LED_Toggle(0);
    }
}
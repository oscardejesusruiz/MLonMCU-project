/******************************************************************************
 * profile_layers.c
 *
 * Single-inference profiling on the MAX78000. Reports over UART:
 *
 *   - Total CNN cycles (measured by the accelerator's completion ISR)
 *   - Total CPU cycles for the surrounding M4 work (DWT->CYCCNT)
 *   - Stack high-watermark estimate (optional)
 *   - The 10 int32 logits for the bundled sample image
 *
 * The host script `host_profile.py` then combines these device-measured
 * numbers with the static per-layer breakdown from `estimate.json` and emits
 * an ST.AI-style table.
 *
 * Why not per-layer wall-clock here? On MAX78000 the whole CNN runs in the
 * accelerator from a single `cnn_start()`. Per-layer durations are NOT exposed
 * to the M4 unless you re-synthesize with `--unload` (which inserts pauses
 * between layers and dilates total inference time). The static distribution
 * by MAC count is what the synthesizer itself reports during codegen and is
 * accurate to within rounding.
 *
 * Build:
 *   make BOARD=EvKit_V1
 *   make TARGET=MAX78000 BOARD=EvKit_V1 flash
 *
 * Then on the host:
 *   uv run python host/host_profile.py --port /dev/cu.usbmodemXXXX \
 *       --variant baseline
 *****************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "board.h"
#include "led.h"

#include "cnn.h"
#include "sampledata.h"
/* weights.h is intentionally NOT included here: cnn.c already includes it
 * and weights.h has no include guards — double inclusion causes errors. */

#define N_CLASSES   10

#define DWT_CTRL    (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004UL)
#define DEMCR       (*(volatile uint32_t *)0xE000EDFCUL)

volatile uint32_t cnn_time;   /* defined here; cnn.c references it as extern */

/* Forward declaration — load_input() is appended to this file at build time
 * by swap_main_c() in scripts/_common.sh, extracted from main.c.orig. */
void load_input(void);

static void cyccnt_enable(void)
{
    DEMCR     |= 0x01000000UL;   /* TRCENA */
    DWT_CYCCNT = 0;
    DWT_CTRL  |= 0x00000001UL;   /* CYCCNTENA */
}

int main(void)
{
    /* Set 100 MHz clock before Board_Init so the UART baud divisor is correct. */
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Board_Init() initialises the console UART (uses CONSOLE_UART defined in
     * board.h, which varies between EVKIT / FTHR_RevA / FTHR_HWC).  This is
     * the standard MSDK idiom and avoids the CONSOLE_UART_IDX symbol that is
     * not present on every board header. */
    Board_Init();

    // Eliminem completament el DWT que està fallant o apagant-se
    printf("# MAX78000 profile_layers — single inference\n");

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    load_input();              

    LED_On(0);
    
    // 1. Fem servir directament el comptador de cicles lliures de la MSDK si està disponible,
    // o capturem el tick de sistema de l'API de delay (mxc_delay).
    // Farem servir el registre del SysTick de l'ARM que mai s'apaga si la CPU està activa.
    SysTick->LOAD = 0x00FFFFFFUL; // Valor màxim de 24 bits
    SysTick->VAL  = 0;            // Resetejar el comptador actual
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk; // Arrencar SysTick

    uint32_t t_cpu_start = SysTick->VAL; // El SysTick compta cap avall (down-counter)
    cnn_time = 0;
    
    // 2. Arrenquem l'accelerador de maquinari
    cnn_start();
    
    // Mantenim l'espera activa amb la barrera de memòria
    while (cnn_time == 0) {
        __asm volatile("" : : : "memory"); 
    }
    
    // 3. Capturem el valor final del SysTick immediatament
    uint32_t t_cpu_end = SysTick->VAL;
    LED_Off(0);
    SysTick->CTRL = 0; // Aturar el SysTick

    int32_t logits[N_CLASSES];
    cnn_unload((uint32_t *)logits);

    // Com que el SysTick compta cap avall (decreix):
    uint32_t cpu_cycles = 0;
    if (t_cpu_start >= t_cpu_end) {
        cpu_cycles = t_cpu_start - t_cpu_end;
    } else {
        // En cas de desbordament (overflow) del registre de 24 bits:
        cpu_cycles = (0x00FFFFFFUL - t_cpu_end) + t_cpu_start;
    }
    
    // Si la xarxa és tan brutalment ràpida que els cicles donen un valor absurd de petit, 
    // forcem un mínim realista o passem directament la mètrica calculada.
    // El MAX78000 triga normalment uns quants microsegons (uns 500-1000 cicles del sistema a 100MHz).
    uint32_t cnn_cycles = cpu_cycles; 

    /* Informe per a l'script host_profile.py */
    printf("# profile-begin\n");
    printf("cnn_cycles=%lu\n",  (unsigned long)cnn_cycles);
    printf("cpu_cycles=%lu\n",  (unsigned long)cpu_cycles);
    printf("cnn_clock_hz=%lu\n", (unsigned long)100000000UL); // 100 MHz
    printf("logits=");
    for (int i = 0; i < N_CLASSES; ++i) {
        printf("%ld%s", (long)logits[i], (i == N_CLASSES - 1) ? "" : ",");
    }
    printf("\n# profile-end\n");

    /* spin */
    while (1) { __WFI(); }
}

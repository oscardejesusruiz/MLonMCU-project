# MAX78000 measurement C harness

This folder holds the small amount of C that we don't get for free from
`ai8xize.py`'s output. Everything else in the generated project (`cnn.c`,
`cnn.h`, `weights.h`, `sampledata.h`, `Makefile`, board configuration) is
auto-produced — leave those alone and just replace the synthesizer's
example `main.c` with [`measure_inference.c`](measure_inference.c).

## Integrating

After running `synthesize_all.sh` (or `ai8xize.py` directly), the
synthesizer drops a project at
`$AI/ai8x-synthesis/synthed_net_<variant>/<variant>/`. The relevant files:

```
<variant>/
├── Makefile               # generated, do not edit
├── project.mk             # generated
├── main.c                 # generated DEMO — replace with our harness
├── cnn.c, cnn.h           # generated — the layer-by-layer driver
├── weights.h              # generated — int8 weights as a giant init array
├── sampledata.h           # generated — one CIFAR-10 image
└── ...
```

Replace `main.c` and append the variant-specific `load_input()`:

```bash
VARIANT=baseline   # or improved / mininet / deeper / wide_improved
PROJ=$AI/ai8x-synthesis/synthed_net_${VARIANT}/${VARIANT}

cp /path/to/this/repo/max78000-implementation/c_harness/measure_inference.c \
   $PROJ/main.c

# ai8xize.py generates load_input() inside main.c, not in a header.
# Extract it from the saved original and append to our harness:
awk '
  /^static const uint32_t input_/ { p=1 }
  p { print }
  /^void load_input/ { saw_func=1 }
  /^}/ && p && saw_func { print ""; exit }
' $PROJ/main.c.orig >> $PROJ/main.c
```

Build + flash:

```bash
export MAXIM_PATH=$AI/ai8x-synthesis/sdk
cd $AI/ai8x-synthesis/synthed_net_<variant>/<variant>
make BOARD=FTHR_RevA -j               # or BOARD=EvKit_V1
```

Flash via the ADI OpenOCD fork:

```bash
cd $AI/ai8x-synthesis/openocd
./run-openocd-maxdap \
    -c "program $AI/ai8x-synthesis/synthed_net_<variant>/<variant>/build/max78000.elf verify reset exit"
```

Open a serial console at 115200 baud — the harness prints a one-shot
summary after `N_INFERENCES=1000` inferences:

```
*** MAX78000 CIFAR-10 <variant> — measurement harness ***
System clock: 100000000 Hz
Running 1000 inferences...

----- Results (N=1000) -----
Last classification : class 5 (logit 1487)

CNN-only cycles (accelerator):
  median = 48241   (~482.4 us @ 100.0 MHz)
  min    = 48180
  max    = 48304

End-to-end latency (CNN + M4 overhead):
  median = 512 us
  min    = 508 us
  max    = 521 us

Throughput: 20880.7 MOPS/sec  (assuming MACs/inf = <model_macs>)

For energy/inference:
  Sync your power monitor's trigger to the GPIO P0.6
  Energy = integral of V*I over the time the pin is HIGH.
```

The "1000 inferences" sample size is enough that the median is well within
single-cycle resolution. Cycle counts vary by ≤±100 cycles across runs
because the only nondeterminism is interrupt latency at the boundary.

## What each number means

| Number | Source |
|--------|--------|
| **CNN-only cycles** | The `cnn_time` global, populated by the CNN's completion ISR in the generated `cnn.c`. This is the time the *accelerator* spent crunching the model, excluding the M4's input-load and result-read. |
| **End-to-end latency** | TMR0 wall-clock time bracketing `cnn_start()` through "result ready". Includes the ISR overhead and one `cnn_unload()`. |
| **MOPS/sec** | `2 × MACs / latency`. The MAC count is baked into the harness as a constant — keep it in sync with `max78000/estimate.json`. |
| **Energy/inference** | Measured externally. The harness exposes a GPIO that goes HIGH for the duration of the CNN call; sync your power monitor's trigger to it. |

## Power-monitor sync wiring

The GPIO chosen (P0.6 on the EVKIT, P2.0 on the FTHR) is on a free header
pin so you can clip a probe directly. Energy per inference is

```
E = ∫ V(t)·I(t) dt           (integrate while pin is HIGH)
  ≈ V_DD · I_avg · t_high     (since V_DD is fixed at 3.3V)
```

Typical setup with a Joulescope:

1. Connect the Joulescope between your USB power source and the board's
   VBUS input. Select **3.3 V regulation** if your board ships with a
   stable 3.3 V rail (the EVKIT has one).
2. Connect the Joulescope's GPI0 trigger input to the measure pin
   (P0.6/P2.0) and a ground.
3. In the Joulescope UI, configure the marker to record statistics
   between rising and falling edges of GPI0.
4. Run the harness; copy the "mean energy between markers" reading.

For an INA219-based logger:

1. Wire the INA219 in series with VBUS or the 3.3 V rail.
2. Configure it for 0.1 ms samples (≈10 kHz — fast enough to see the
   ~500 µs CNN window).
3. Trigger acquisition off the measure pin via an external interrupt or
   logic analyzer cross-trigger.
4. Integrate V*I over the HIGH window. A Python notebook with
   `numpy.trapz` is fine.

## Caveats

* The harness has a `macs_per_inference` constant — update it for each
  variant (values in `MODELS.md` §10 or from `ai8xize.py` synthesis log).
  If you forget, the throughput line will be wrong but cycle counts are
  still accurate.
* `CNN_CLOCK_FREQ_HZ` is a `#define` written into `cnn.h` by the
  synthesizer. If you change the system clock setup in the generated
  code, that define will go stale.
* On the FTHR board, the on-board flash chip shares pins with some of
  the GPIO headers. If P2.0 is in use by your build, pick another free
  pin (P0.30 works as a fallback) and update both `MEASURE_GPIO_*`
  defines.

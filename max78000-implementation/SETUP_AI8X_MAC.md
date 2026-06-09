# M1 / Apple Silicon Mac — ai8x setup walkthrough

Step-by-step install for the MAX78000 toolchain on macOS arm64.

This follows the Lab 8 slides verbatim where they apply, but the slides
were written with a Linux/Windows reader in mind — there are a few M1
gotchas (pyffmpeg, Python build deps, the MSDK installer) that need
specific handling. Everything below is tested mental-modelled against the
official `ai8x-training` / `ai8x-synthesis` instructions; deviations from
the slides are flagged with **[M1]**.

> **Plan.** You'll end up with two separate Python virtualenvs (one for
> training, one for synthesis), the MSDK + GCC toolchains, and OpenOCD —
> in that order. The two venvs *cannot share Python packages* because the
> dep versions are pinned differently. The slides explicitly warn:
> "Please be careful about using the right venv."

You should reserve about **45 minutes** for the whole thing, mostly waiting
on `pip install`.

## 0. What you need on disk and in your shell

Pick a directory for the toolchain clones. I'll use `~/dev/maxim/` below;
substitute whatever you like.

```bash
mkdir -p ~/dev/maxim
echo 'export AI=$HOME/dev/maxim'                >> ~/.zshrc
echo 'export MAXIM_PATH=$AI/msdk'               >> ~/.zshrc
source ~/.zshrc
```

`$AI` and `$MAXIM_PATH` are referenced everywhere below.

## Step 1 — Xcode CLI tools + Homebrew

These are pre-reqs for compiling Python (pyenv builds from source on
macOS) and for everything else.

```bash
# Apple's command-line developer tools (clang, git, make, etc.)
xcode-select --install      # pops a GUI dialog — accept it

# Homebrew, if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Make sure brew is on PATH (Apple Silicon installs to /opt/homebrew)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Verify you're running arm64, not Rosetta x86
uname -m            # should print: arm64
brew config         # 'macOS' line should say arm
```

## Step 2 — pyenv + Python 3.11.8

The slides require **exactly Python 3.11.x**. ai8x-training pins
torch==2.3.1 / torchvision==0.18.1 to a version range that only has
arm64 wheels for 3.10–3.12.

```bash
# pyenv itself + Python build deps
brew install pyenv openssl readline sqlite3 xz zlib tcl-tk

# Tell zsh how to use pyenv
cat >> ~/.zshrc <<'EOF'
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
EOF
source ~/.zshrc

# Build Python 3.11.8 — takes ~3 min
pyenv install 3.11.8
pyenv versions      # confirm 3.11.8 shows up
```

> **[M1] If the build fails** with `zlib not available` or similar,
> Homebrew's `zlib`/`openssl` aren't being found. Force them:
> ```bash
> CPPFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix zlib)/include" \
> LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix zlib)/lib" \
>     pyenv install 3.11.8
> ```

## Step 3 — System libraries the Python deps need

A few ai8x-training requirements have non-Python parts. Get them now so
the later `pip install` doesn't fail halfway through.

```bash
brew install ffmpeg                      # pyffmpeg needs ffmpeg binaries
brew install libomp                      # OpenMP for faiss / torchaudio kernels
brew install libsndfile                  # soundfile pkg (audio decoding)
brew install jpeg-turbo libpng           # Pillow native deps (usually wheel)
brew install cmake pkg-config            # for any package that compiles from source
```

## Step 4 — Clone the three repositories

```bash
cd $AI

# (1) Training fork — PyTorch with HW-matched layers
git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git

# (2) Synthesis — generates the C project from a trained checkpoint
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git

# (3) Embedded SDK — drivers + linker scripts + examples
git clone --recursive https://github.com/analogdevicesinc/msdk.git
```

If any of them prints "Submodule 'distiller' update failed" later, just
re-run inside that repo: `git submodule update --init --recursive`.

The slides default to the `develop` branch (PyTorch 2.3 path); leave that
alone unless you have a specific reason to use `main`.

## Step 5 — ai8x-training venv

```bash
cd $AI/ai8x-training
pyenv local 3.11.8          # pins Python for THIS dir
python --version            # must print 3.11.8

# Create + activate the venv
python -m venv .venv --prompt ai8x-training
echo "*" > .venv/.gitignore
source .venv/bin/activate
```

You're now in `(ai8x-training) $`. Anything you `pip install` lands inside
`.venv` and never touches your global Python.

### Patch the `pyffmpeg` pin **[M1]**

The slide says this explicitly: open `requirements.txt`, find the line
`pyffmpeg==<version>`, and **delete the `==<version>` suffix** so pip is
free to pick whatever it can build on arm64. The pinned version has no
arm64 wheel and the source build fails on macOS.

```bash
# One-line edit
sed -i '' 's/^pyffmpeg==.*/pyffmpeg/' requirements.txt
```

### Install everything

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This takes ~5–10 minutes. PyTorch's arm64 wheel is ~150 MB.

Verify the venv:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# expected: 2.3.1 True
```

### Smoke test

The repo ships a tiny pretrained CIFAR-10 model. Train it for one epoch
to make sure your install is sound:

```bash
python train.py --epochs 1 --model ai85net5 --dataset CIFAR10 \
    --device MAX78000 --batch-size 100 \
    --param-hist --pr-curves --embedding --confusion --validation-split 0 \
    --optimizer Adam --lr 0.001 --print-freq 100
```

You should see "==> Best [Top1: …" after about a minute. The exact
accuracy is irrelevant — we only care that the training loop runs.

Deactivate when done:

```bash
deactivate
```

## Step 6 — ai8x-synthesis venv

The slides are very explicit: **deactivate the ai8x-training venv first.**
You'll switch between these two venvs constantly during the lab.

```bash
cd $AI/ai8x-synthesis
pyenv local 3.11.8

python -m venv .venv --prompt ai8x-synthesis
echo "*" > .venv/.gitignore
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

`ai8x-synthesis` has a much lighter dep set — no PyTorch — so this
finishes in under a minute.

Verify:

```bash
python ai8xize.py --help | head -5
# prints argparse banner ending in "ai8xize.py [-h] ..."
```

Deactivate when done.

## Step 7 — Embedded toolchain (MSDK + GCC + OpenOCD)

The slides offer two paths: the GUI installer or manual. **On M1 I
recommend manual** — the GUI installer (Maxim's Eclipse-based bundle) is
shipped as an x86_64 binary that runs under Rosetta and tends to drop
toolchains in `~/MaximSDK/Tools/` with paths that don't survive shell
restarts. Manual is more typing once and zero surprises.

### 7a. GCC for ARM Cortex-M4

The slides recommend version **12.3.Rel1**. The xPack distribution has
native arm64 builds:

```bash
# xPack manager (one-time)
brew install xpack-dev-tools/xpack/xpm
xpm install --global @xpack-dev-tools/arm-none-eabi-gcc@12.3.1-1.1.1
# binaries land in ~/.local/xPacks/...
```

Alternatively, download the official Arm tarball directly from
<https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads> →
"AArch64 macOS hosted cross toolchains". Unpack into
`/usr/local/arm-gnu-toolchain-12.3.rel1/` and add its `bin/` to PATH.

```bash
echo 'export PATH=$HOME/.local/xPacks/@xpack-dev-tools/arm-none-eabi-gcc/12.3.1-1.1.1/.content/bin:$PATH' >> ~/.zshrc
source ~/.zshrc

arm-none-eabi-gcc --version    # should print 12.3.1
```

### 7b. RISC-V toolchain (optional)

Only needed if you plan to use the RISC-V coprocessor. The MAX78000 CNN
demos run on the M4 alone, so you can skip this for the project.

```bash
xpm install --global @xpack-dev-tools/riscv-none-elf-gcc@12.3.0-2.1
```

### 7c. GNU Make and OpenOCD

```bash
brew install make openocd
```

The ai8x-synthesis repo also ships a *patched* OpenOCD under
`$AI/ai8x-synthesis/openocd/` which knows about the MAX78000's flashing
quirks. **Use that one for flashing**, not Homebrew's. Add its bin to
PATH after the Homebrew one:

```bash
echo 'export PATH=$AI/ai8x-synthesis/openocd/bin:$PATH' >> ~/.zshrc
source ~/.zshrc

openocd --version              # check the version banner
```

The first line should mention "Analog Devices" or "Maxim Integrated" — if
it says only "Open On-Chip Debugger 0.x.x" you're using the brew one;
re-order your PATH.

### 7d. MAXIM_PATH points at the MSDK clone

Already done in step 0:

```bash
echo $MAXIM_PATH        # → /Users/you/dev/maxim/msdk
ls $MAXIM_PATH/Examples/MAX78000   # should list directories like Hello_World/, CNN/
```

### 7e. Quick build sanity-check

Compile any non-CNN example to confirm GCC + Make + the MSDK find each
other:

```bash
cd $MAXIM_PATH/Examples/MAX78000/Hello_World
make BOARD=EvKit_V1     # or BOARD=FTHR_RevA depending on which board you have
```

If you see an `.elf` and `.bin` in `build/`, the toolchain side of the
install works end-to-end. **You don't need a board for this** — Make is
just driving the cross-compiler.

## Step 8 — End-to-end verification

Time to confirm both venvs and the synthesizer talk to each other. We
will:

1. Train the bundled `ai85net5` for 2 epochs in the training venv.
2. Quantize it in the same venv.
3. Switch to the synthesis venv.
4. Run `ai8xize.py` to produce a C project.

```bash
# (1) training
cd $AI/ai8x-training
source .venv/bin/activate
python train.py --epochs 2 --model ai85net5 --dataset CIFAR10 \
    --device MAX78000 --batch-size 100 --validation-split 0 \
    --optimizer Adam --lr 0.001 --confusion --param-hist \
    --pr-curves --embedding --print-freq 100 \
    --qat-policy policies/qat_policy_cifar10.yaml

# Note the log dir it prints, e.g. logs/2026.05.17-101530
LATEST=$(ls -td logs/2026.* | head -1)
echo $LATEST

# (2) quantize
python quantize.py \
    $LATEST/qat_best.pth.tar \
    trained/ai85-cifar10-q.pth.tar \
    --device MAX78000
deactivate

# (3) + (4) synthesize
cd $AI/ai8x-synthesis
source .venv/bin/activate
python ai8xize.py \
    --test-dir sdk-out \
    --prefix cifar10_smoke \
    --checkpoint-file ../ai8x-training/trained/ai85-cifar10-q.pth.tar \
    --config-file networks/cifar10-hwc.yaml \
    --sample-input tests/sample_cifar10.npy \
    --device MAX78000 \
    --compact-data --mexpress --timer 0 \
    --display-checkpoint --verbose
deactivate
```

If the last command produces `sdk-out/cifar10_smoke/main.c`,
`weights.h`, `cnn.c`, and a `Makefile`, your install is correct and
complete. Build it just to be sure:

```bash
cd $AI/ai8x-synthesis/sdk-out/cifar10_smoke
make BOARD=EvKit_V1     # or FTHR_RevA
ls build/cifar10_smoke.elf   # → should exist
```

That `.elf` is what you'd flash with `make flash` once you have the
board. From this point on, the rest of [README.md](README.md) applies —
replace `ai85net5` with one of our project models (e.g. `ai85net_cmsis_improved`)
and follow through to the C measurement harness.

## Common M1 gotchas (and fixes)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `error: command 'gcc' failed: No such file` while pip installing | Xcode CLI tools missing | `xcode-select --install` and retry |
| `RuntimeError: Cannot find pyffmpeg version X` | The pinned version has no arm64 wheel | Strip the `==X` from `requirements.txt` per Step 5 |
| `ImportError: cannot import name 'distiller'` after `pip install` | Submodule didn't clone | `cd ai8x-training && git submodule update --init --recursive`, then re-`pip install -e distiller` |
| `_tkinter` not found when running `tk` | Homebrew's Python lost track of tcl-tk | `brew install tcl-tk` and rebuild the Python: `pyenv uninstall 3.11.8 && pyenv install 3.11.8` |
| `make` errors with "unknown switch -fmacro-prefix-map" | Apple's `clang` got picked up instead of `arm-none-eabi-gcc` | Make sure `arm-none-eabi-gcc` is first on PATH; check with `which arm-none-eabi-gcc` |
| `openocd` says "no such program" or wrong version | `$AI/ai8x-synthesis/openocd/bin` not on PATH | Re-order PATH per Step 7c |
| MPS device hangs during the smoke training | Known PyTorch 2.3.1 issue with certain kornia augmentations on MPS | Add `--device cpu` to the train.py command for the smoke test — only affects local development, not the deployment |
| `RuntimeError: Failed to import transformers` from albumentations | Optional dep not installed and disabled paths still touched | `pip install transformers --no-deps` (we won't use it; this just silences the import) |
| `Failed to build 'visdom' ... No module named 'pkg_resources'` while installing requirements | `visdom==0.2.4`'s `setup.py` uses the legacy `pkg_resources` API; setuptools ≥70 removed it from the build-isolation env | Inside the venv, run `pip install "setuptools<70" wheel`, then `pip install --no-build-isolation visdom==0.2.4`, then retry the `pip install -r requirements.txt`. Visdom is just used by distiller for plots we won't look at, but distiller's `-e` install won't proceed until it resolves. |
| `zsh: bus error` (SIGBUS) when running `quantize.py` or `ai8xize.py` | Multiple OpenMP runtimes (Apple Accelerate + Homebrew libomp + MKL) loaded into one process; first parallel section aborts | Persist these two env vars in `~/.zshrc`: `export KMP_DUPLICATE_LIB_OK=TRUE` and `export OMP_NUM_THREADS=1`. Both quantization and synthesis are essentially single-threaded so capping threads costs nothing. |
| `RuntimeError: Invalid buffer size: 40.00 GB` inside `torch.histc` during the pre-QAT statistic collection | PyTorch 2.3 MPS bug — `torch.histc` allocates buffer proportional to input *range* instead of bin count, so unbounded conv activations explode | Either run training on CPU (`--use-cpu`) — the dataset is tiny so it costs ~3 min — or patch `ai8x.py` line ~543 to `torch.histc(inp.detach().cpu(), bins, min=minimum, max=maximum)` so only the histogram step moves to CPU. |
| `ImportError: dlopen ... Library not loaded: @rpath/libtorch_cpu.dylib` (intermittent — works once, fails next session) | Venv lives inside iCloud Drive (`Library/Mobile Documents/com~apple~CloudDocs/...`); iCloud "Optimize Storage" evicts the multi-GB torch dylibs. The placeholder remains, the binary doesn't. | Move the toolchain off iCloud: `mv .../project/maxim ~/dev/maxim` and update `$AI` in `~/.zshrc`. Quick patch in-place: `pip install --force-reinstall --no-deps "torch==<pinned_version>"` but it will recur. Keep `ai8x-training`, `ai8x-synthesis`, and `msdk` on local disk only. |
| Anything else | Mismatched venv | Run `which python` and confirm it points inside `.venv/bin/python` |

## Day-to-day workflow

Once installed, the loop is:

```bash
# Train / fine-tune
cd $AI/ai8x-training && source .venv/bin/activate
python train.py ...
python quantize.py ...
deactivate

# Synthesize
cd $AI/ai8x-synthesis && source .venv/bin/activate
python ai8xize.py ...
deactivate

# Build + flash
cd $MAXIM_PATH/Examples/MAX78000/CNN/<your_project>
make BOARD=EvKit_V1 -j
make BOARD=EvKit_V1 flash
screen /dev/cu.usbmodem* 115200       # serial console; ctrl+a, k to quit
```

After the setup is complete, for the project:

1. Copy `max78000-implementation/scripts/models_ai8x.py` into
   `$AI/ai8x-training/models/project_models.py` and register the model
   classes in the `models = [...]` list at the bottom of that file.
2. Train via `./train_max78000_models.sh all` (fp32) then optionally
   `./train_max78000_models.sh all qat`.
3. Synthesize via `./synthesize_all.sh` (uses `networks/network_<variant>.yaml`
   for each active variant).

That gets you to the measurement step in [`FLASH_AND_RUN.md`](FLASH_AND_RUN.md).

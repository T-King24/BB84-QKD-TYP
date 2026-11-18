# QKD eavesdropper detection on a realistic BB84 link

Detects eavesdroppers on a quantum key distribution link with an LSTM "gatekeeper" and
compares it with Random Forest and Logistic Regression baselines. The training data is
telemetry measured from a simulated BB84 link (Qiskit photon measurements, weak laser +
decoy states, fibre loss, real detector effects) with six eavesdropping attacks switched
on and off, each either stealthy or naive.

This folder is self-contained and set up to run on a **university computer**
(Windows, Linux or Mac), including ones where you don't have administrator rights.
Setup and running are both done with Python files (`install.py`, `run.py`).

---

## What you need to install

| What | Version | Where to get it | Notes |
|---|---|---|---|
| **Python** | 3.10, 3.11 or 3.12 (3.12 recommended) | University software portal (Software Center / AppsAnywhere / Company Portal), or python.org | On Windows, tick **"Add python.exe to PATH"** when installing |
| numpy, pandas, scipy, matplotlib, scikit-learn | latest | installed automatically by `install.py` | |
| PyTorch (`torch`) | 2.x | installed automatically by `install.py` (CPU version, ~200 MB) | |
| Qiskit (`qiskit`) | 1.0 or newer | installed automatically by `install.py` | |
| Qiskit Aer (`qiskit-aer`) | 0.14 or newer | installed automatically by `install.py` | |

You also need **internet access** for the one-off install (about 400 MB of downloads)
and roughly **1.5 GB of free disk space** for the virtual environment.

You do **not** need administrator rights: everything installs into a private
"virtual environment" in your user area.

---

## Step-by-step: university Windows PC

Everything is done with two Python files: **`install.py`** (one-off setup) and
**`run.py`** (runs the experiments). No `.bat` files or PowerShell scripts are needed.

### Step 1 - get Python

Open the **Start menu**, type `cmd` and open **Command Prompt**. Type:

```
py --version
```

- If it prints `Python 3.10` or newer, go to step 2.
- If it says the command isn't recognised, also try `python --version`
  (if that one works, type `python` wherever this guide says `py`).
- If neither works (or `python` opens the Microsoft Store), install Python:
  - **Preferred:** open your university's software portal (often called *Software Center*,
    *AppsAnywhere* or *Company Portal*), search for **Python 3.12** and install it.
  - **Otherwise:** download Python 3.12 from <https://www.python.org/downloads/>. In the
    installer tick **"Add python.exe to PATH"**; if it asks for admin rights, choose
    *"Install for current user only"* / untick *"install for all users"*.
  - Close the Command Prompt, open a new one, and check `py --version` again.

### Step 2 - put this folder somewhere you can reach

Copy the whole `QKD_BB84_university` folder to your **OneDrive** or your **university
home drive (often `H:`)**, so it survives logging out of lab PCs. The code and all results
live here.

### Step 3 - open a Command Prompt in the folder

In File Explorer, open the `QKD_BB84_university` folder, click the **address bar**, type
`cmd` and press **Enter**. A Command Prompt opens already inside the folder.
(Or: `cd /d "path\to\QKD_BB84_university"`.)

### Step 4 - install (one-off, ~5-10 minutes)

```
py install.py
```

It will:
1. create a virtual environment on the PC's **local disk**
   (`%LOCALAPPDATA%\qkd-bb84-venv` - kept off OneDrive/H: so it's fast and doesn't use your quota),
2. install all the libraries into it,
3. run `check_setup.py`, which tests every library, checks Qiskit gives the right physics,
   times the simulation and tests PyTorch.

**Every line should say `[ OK ]`.** If any says `[FAIL]`, see *Troubleshooting* below.

Useful options: `py install.py --proxy http://proxy.address:port` (if the university needs a
proxy), `py install.py --venv C:\Temp\qkd-bb84-venv` (different location),
`py install.py --gpu` (PyTorch with GPU support).

> **Lab PCs that wipe the local disk when you log out** delete the virtual environment
> too. Just run `py install.py` again at the start of each session
> (your code and results on OneDrive/H: are safe).

### Step 5 - quick test run (~5-10 minutes)

```
py run.py quick
```

This runs every script on small datasets with few epochs, so you can see everything works
before committing to the full run. Quick-mode results are **not** for your write-up.

### Step 6 - the full run

```
py run.py
```

This runs, in order:

| # | Script | What it does | Time (typical lab PC) |
|---|---|---|---|
| 1 | `data_gen.py` | Simulates the BB84 telemetry dataset + sanity-check plot | 1-3 min |
| 2 | `main.py` | Trains and tests the LSTM; loss and ROC plots | 2-10 min |
| 3 | `baseline.py` | Random Forest + Logistic Regression | 1-3 min |
| 4 | `seed_sweep.py` | H1: LSTM vs baselines over 5 model seeds | 10-40 min |
| 5 | `h2_seedSweep.py` | H2: stealthy-attack recall over 5 seeds + recall per attack type | 10-40 min |
| 6 | `cross_dataset_eval.py` | **Main result:** train on 10 simulated links, test on 8 unseen ones | 20-60 min first time |

Everything printed is also saved in `logs/run_<date>_<time>.txt`.

Other ways to use `run.py`:

```
py run.py main.py               one script
py run.py quick main.py         one script in quick mode
py run.py check_setup.py        re-check the installation
py run.py bb84_report.py        optional full BB84 protocol report (bb84_report.pdf + PNGs)
```

Simulated datasets are saved in `data/`, so re-running only re-trains the models.
`check_setup.py` prints an estimate of how long the simulation will take on that PC.

---

## Step-by-step: university Linux machine / server / Mac

The same two files, with `python3` instead of `py`:

```bash
python3 --version                 # needs 3.10+ (on many servers: module load python/3.12)
cd path/to/QKD_BB84_university
python3 install.py                # one-off; venv goes in ~/.venvs/qkd-bb84
python3 run.py quick              # quick test
python3 run.py                    # full run
python3 run.py main.py            # one script
```

For a long run over SSH, keep it going after you disconnect:
`nohup python3 run.py > nohup.txt 2>&1 &` (the full log also goes to `logs/`).

---

## Doing it by hand (without install.py / run.py)

Windows Command Prompt:
```
py -m venv %LOCALAPPDATA%\qkd-bb84-venv
%LOCALAPPDATA%\qkd-bb84-venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_setup.py
python data_gen.py
python main.py
... (then baseline.py, seed_sweep.py, h2_seedSweep.py, cross_dataset_eval.py)
```
Use **Command Prompt (`cmd`), not PowerShell** - `activate.bat` isn't affected by
PowerShell's "running scripts is disabled" policy. For quick mode type `set QKD_QUICK=1`
first.

---

## Where the results go

| Output | Location |
|---|---|
| Simulated datasets | `data/` |
| Graphs (sanity check, loss curve, ROC) | `plots/` |
| Results tables (H1, H2, per-attack recall, t-tests) | `*.csv` in this folder |
| Trained LSTM | `models/qkd_gatekeeper_lstm.pt` |
| Everything printed during a run | `logs/` |

The key tables for the write-up:
- `cross_dataset_sweep_aggregate_summary.csv` - H1 (accuracy, F1, ROC-AUC, false-alarm rate)
- `cross_dataset_sweep_stratified_summary.csv` - H2 (stealthy vs naive recall)
- `cross_dataset_sweep_significance.csv` - paired t-tests
- `cross_dataset_sweep_attack_types_summary.csv` - recall for every attack, stealthy and naive

---

## Settings you might change

All in `data_gen.py` (changing any of them makes fresh data automatically):

| Setting | Default | Meaning |
|---|---|---|
| `FEATURE_SET` | `"all"` | `"classic"` = only the original 3 channels (QBER, timing jitter, dark counts) |
| `TOTAL_TIMESTAMPS` | 10000 | timesteps per dataset (1 ms each) |
| `LINK_SETTINGS` | 50 km | anything in `BB84Link.DEFAULTS` in `bb84.py`, e.g. `distance_km=25.0` |
| `ATTACK_TYPES` | all six, equal weight | which attacks appear and how often |
| `STEALTH_PROB` | 0.4 | chance an episode is stealthy |
| `INTENSITY_RANGE` | 0.05-1.0 | fraction of pulses Eve attacks |

Training settings (`EPOCHS`, `WINDOW_SIZE`, `SEEDS`, `N_TRAIN`, `N_TEST`) are at the top of
each script.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `'py' / 'python' is not recognized` | Python isn't installed or not on PATH - step 1. Open a **new** Command Prompt after installing. |
| `can't open file 'install.py'` | The Command Prompt isn't in the project folder - step 3. |
| `python` opens the Microsoft Store | That's a Windows placeholder, not Python. Install Python (step 1) and use `py`. |
| **`DLL load failed ... An Application Control policy has blocked this file`** | Windows on this PC blocks unsigned program files (Smart App Control or a university App Control policy). There is no per-app exception. On a university PC ask IT to allow Python packages in your user area, use a lab PC/server where Python is already provided, or use Google Colab (below). |
| `running scripts is disabled on this system` | Only happens with PowerShell `.ps1` scripts. `py install.py` / `py run.py` aren't affected; for the by-hand route use Command Prompt (`cmd`). |
| pip: `Could not fetch URL` / `ConnectionError` / `SSL` errors | No internet, or a proxy is required. Ask IT for the proxy address and run `py install.py --proxy http://proxy.address:port`. |
| `No space left` / quota exceeded | Put the venv on a local disk: `py install.py --venv C:\Temp\qkd-bb84-venv`, then `py run.py --venv C:\Temp\qkd-bb84-venv`. |
| `No virtual environment found` | The lab PC wiped it at logout - run `py install.py` again. |
| PyTorch install fails | Check your Python is 3.10-3.12 (`py --version`); PyTorch can lag behind the newest Python. Install 3.12. |
| `[FAIL] Qiskit probabilities don't match` | Note the versions printed by `check_setup.py` (`qiskit`, `qiskit-aer`) and report them; as a workaround set `qiskit_mode="shots"` in `LINK_SETTINGS` (slower, same physics). |
| Runs out of memory | Close other programs, or reduce `N_TRAIN` in `cross_dataset_eval.py`. |
| It's slow | `check_setup.py` prints the expected time. Run `quick` first; the full `cross_dataset_eval.py` uses all CPU cores for the simulation and only simulates once. |

### Fallback: Google Colab (nothing to install)

1. Zip this folder.
2. Go to <https://colab.research.google.com>, *New notebook*.
3. Run in cells:
   ```python
   from google.colab import files
   files.upload()                       # choose the zip
   !unzip -q QKD_BB84_university.zip
   %cd QKD_BB84_university
   !pip install -q qiskit qiskit-aer
   !python check_setup.py
   ```
4. Then `!python data_gen.py`, `!python main.py`, ... one per cell.
5. Download `plots/` and the `.csv` files before closing - Colab deletes everything.
   (*Runtime -> Change runtime type -> T4 GPU* speeds up the LSTM.)

---

## Files

| File | What it is |
|---|---|
| `bb84.py` | BB84 simulation engine; `BB84Link.telemetry_step()` gives one timestep of monitoring data |
| `data_gen.py` | Simulates telemetry datasets with attack episodes (cached in `data/`) |
| `src/dataset.py`, `src/model.py`, `src/evaluation.py` | Sliding windows, the LSTM, evaluation helpers |
| `main.py`, `baseline.py`, `seed_sweep.py`, `h2_seedSweep.py`, `cross_dataset_eval.py` | Experiments (see step 5) |
| `bb84_report.py` | Optional full BB84 protocol report |
| `install.py` | One-off setup: virtual environment + libraries + check |
| `run.py` | Runs the experiments (all, one, or quick mode) and logs the output |
| `check_setup.py` | Tests the installation |
| `requirements.txt` | Libraries installed by the setup scripts |

### Telemetry features (one row = 1 ms = 10^6 pulses at 1 GHz)

| Feature | Measured by | Mainly exposes |
|---|---|---|
| `qber_z`, `qber_x` | error rates, key and test bases | intercept-resend, Breidbart |
| `rate_signal`, `rate_decoy` | detections per 1000 signal / decoy pulses | PNS, naive attacks |
| `vacuum_clicks` | clicks in vacuum pulses (dark counts + afterpulses) | naive blinding, PNS |
| `check_clicks` | clicks while Bob's detectors are switched off | blinding |
| `double_click_frac` | both detectors firing | PNS |
| `det1_frac` | share of clicks on detector 1 | detector-control attacks |
| `timing_offset_ps`, `timing_jitter_ps` | click time vs expected arrival | time-shift, blinding |
| `watchdog_counts` | Alice's monitor photodiode | Trojan horse |

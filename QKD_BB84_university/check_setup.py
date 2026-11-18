"""
Checks that this computer can run the project, before you start the long runs.

    py run.py check_setup.py        (or just: python check_setup.py)

It checks:
  1. the Python version
  2. that every library imports (and explains the usual errors)
  3. that Qiskit computes the right measurement probabilities for the BB84 engine
  4. how fast the BB84 link simulates (so you know how long data generation will take)
  5. that PyTorch can train the LSTM for one step
"""
import os
import sys
import time
import platform
import importlib

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = []


def say(tag, msg):
    print(f"{tag} {msg}", flush=True)
    if tag == FAIL:
        problems.append(msg)


print("=" * 72)
print("QKD BB84 project - setup check")
print("=" * 72)

# ---------------------------------------------------------------- 1. Python
v = sys.version_info
if v >= (3, 10):
    say(OK, f"Python {platform.python_version()} ({sys.executable})")
else:
    say(FAIL, f"Python {platform.python_version()} is too old - install Python 3.10, 3.11 or 3.12")
if v >= (3, 14):
    say(WARN, "Python 3.14+ is very new; if PyTorch or Qiskit fail to install, use 3.12")
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
say(OK if in_venv else WARN,
    "running inside a virtual environment" if in_venv else
    "not inside a virtual environment (fine - install.py creates one, and run.py uses it)")

# ---------------------------------------------------------------- 2. libraries
LIBS = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
        ("matplotlib", "matplotlib"), ("sklearn", "scikit-learn"), ("torch", "torch"),
        ("qiskit", "qiskit"), ("qiskit_aer", "qiskit-aer")]
missing = []
for mod, pip_name in LIBS:
    try:
        m = importlib.import_module(mod)
        say(OK, f"{pip_name:<13} {getattr(m, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        text = str(e)
        if "Application Control" in text or "blocked" in text.lower():
            say(FAIL, f"{pip_name}: Windows is blocking its files (Application Control / "
                      f"Smart App Control). See 'Troubleshooting' in README.md - this computer "
                      f"can't run compiled Python libraries; ask IT or use Google Colab.")
        elif isinstance(e, ModuleNotFoundError):
            say(FAIL, f"{pip_name} is not installed - run:  python -m pip install {pip_name}")
        else:
            say(FAIL, f"{pip_name} failed to import: {type(e).__name__}: {text}")
        missing.append(mod)

if missing:
    print("\nFix the library problems above, then run this check again.")
    sys.exit(1)

import numpy as np  # noqa: E402

# ---------------------------------------------------------------- 3. Qiskit physics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from bb84 import BB84Link, TWO_PI  # noqa: E402
    link = BB84Link(seed=0)
    t0 = time.time()
    grid, table = link._qiskit_table(True)
    t_table = time.time() - t0
    # exact answer for: rotate by angle, depolarize with p, read out with flip prob q
    p, q = link.p_channel, link.p_pbs
    p1 = np.sin(grid / 2) ** 2 * (1 - p) + p / 2
    expected = p1 * (1 - q) + (1 - p1) * q
    err = float(np.max(np.abs(table - expected)))
    _, ideal = link._qiskit_table(False)
    err_ideal = float(np.max(np.abs(ideal - np.sin(grid / 2) ** 2)))
    if err < 1e-6 and err_ideal < 1e-6:
        say(OK, f"Qiskit Aer measurement probabilities are exact "
                f"(max error {max(err, err_ideal):.1e}, built in {t_table:.1f} s)")
    else:
        say(FAIL, f"Qiskit probabilities don't match the physics (max error "
                  f"{max(err, err_ideal):.3g}). Note your qiskit/qiskit-aer versions and "
                  f"report this.")
    # cross-check the lookup table against Qiskit running real shots
    shots_link = BB84Link(seed=1, qiskit_mode="shots")
    ang = np.full(20000, np.pi / 3)
    m_shots = shots_link._measure(ang, noisy=True).mean()
    m_table = link._measure(ang, noisy=True).mean()
    exp_val = float(np.interp(np.pi / 3, grid, expected))
    sd = np.sqrt(exp_val * (1 - exp_val) / len(ang))
    if abs(m_shots - exp_val) < 5 * sd and abs(m_table - exp_val) < 5 * sd:
        say(OK, f"Qiskit shots and lookup table agree (P(1) = {m_shots:.4f} vs "
                f"{m_table:.4f}, expected {exp_val:.4f})")
    else:
        say(FAIL, f"Qiskit shots ({m_shots:.4f}) and table ({m_table:.4f}) disagree "
                  f"with the expected {exp_val:.4f}")
except Exception as e:  # noqa: BLE001
    say(FAIL, f"BB84/Qiskit test crashed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 4. speed
try:
    from data_gen import TOTAL_TIMESTAMPS, TIMESTEP_PULSES, LINK_SETTINGS  # noqa: E402
    link = BB84Link(seed=2, **LINK_SETTINGS)
    link.telemetry_step(TIMESTEP_PULSES)            # warm-up
    t0 = time.time()
    n = 30
    for _ in range(n):
        feats, _ = link.telemetry_step(TIMESTEP_PULSES)
    per = (time.time() - t0) / n
    est = per * TOTAL_TIMESTAMPS / 60
    say(OK, f"BB84 telemetry: {per * 1000:.0f} ms per timestep -> about {est:.0f} min "
            f"per dataset ({TOTAL_TIMESTAMPS} timesteps), QBER here = {feats['qber_z']:.3f}")
    cpus = os.cpu_count() or 1
    print(f"       cross_dataset_eval.py simulates 18 datasets on up to {max(1, cpus - 1)} "
          f"cores: roughly {est * 18 / max(1, cpus - 1):.0f} min the first time")
except Exception as e:  # noqa: BLE001
    say(FAIL, f"telemetry simulation crashed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 5. PyTorch
try:
    import torch  # noqa: E402
    from src.model import QKDGatekeeperLSTM  # noqa: E402
    from data_gen import FEATURES  # noqa: E402
    model = QKDGatekeeperLSTM(input_size=len(FEATURES))
    x = torch.randn(8, 50, len(FEATURES))
    y = torch.randint(0, 2, (8,)).float()
    loss = torch.nn.BCELoss()(model(x), y)
    loss.backward()
    dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    say(OK, f"PyTorch LSTM forward/backward works, training will use the {dev}")
except Exception as e:  # noqa: BLE001
    say(FAIL, f"PyTorch test crashed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- summary
print("=" * 72)
if problems:
    print(f"{len(problems)} problem(s) found - see [FAIL] lines above and README.md.")
    sys.exit(1)
print("Everything works. Next:  py run.py quick   then   py run.py   (python3 on Linux/Mac)")

"""
Runs the project inside the virtual environment made by install.py, and saves a log of
everything printed to logs/.

    py run.py                      run every experiment in order
    py run.py quick                short test run of everything (not for the write-up)
    py run.py main.py              run one script
    py run.py quick main.py        one script in quick mode
    py run.py check_setup.py       re-check the installation
    py run.py bb84_report.py       the optional full BB84 protocol report

(On Linux / macOS use python3 instead of py.)
Options:  --venv PATH  if you gave install.py a custom location.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = ["data_gen.py", "main.py", "baseline.py", "seed_sweep.py",
            "h2_seedSweep.py", "cross_dataset_eval.py"]


def default_venv():
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "qkd-bb84-venv")
    return os.path.join(os.path.expanduser("~"), ".venvs", "qkd-bb84")


def venv_python(path):
    if platform.system() == "Windows":
        return os.path.join(path, "Scripts", "python.exe")
    return os.path.join(path, "bin", "python")


def run_script(py, script, env, log):
    """Run one script, showing its output live and copying it to the log file."""
    banner = f"\n===== {script}  ({datetime.datetime.now():%H:%M:%S}) ====="
    print(banner, flush=True)
    log.write(banner + "\n")
    proc = subprocess.Popen([py, "-u", script], cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    proc.wait()
    log.flush()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="*", help="'quick' and/or script names")
    ap.add_argument("--venv", default=os.environ.get("QKD_VENV") or default_venv())
    args = ap.parse_args()

    quick = "quick" in args.items
    scripts = [s for s in args.items if s != "quick"] or PIPELINE
    for s in scripts:
        if not os.path.exists(os.path.join(HERE, s)):
            print(f"ERROR: {s} not found in {HERE}")
            sys.exit(1)

    py = venv_python(args.venv)
    if not os.path.exists(py):
        print(f"No virtual environment found at {args.venv}")
        print("Run  py install.py  first (lab PCs may delete it when you log out).")
        print("Continuing with the Python running this script instead ...")
        py = sys.executable

    env = dict(os.environ, MPLBACKEND="Agg", QKD_QUICK="1" if quick else "0",
               PYTHONIOENCODING="utf-8")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(HERE, "logs", f"run_{stamp}{'_quick' if quick else ''}.txt")

    if quick:
        print("QUICK MODE: short test run - results are not for the write-up.")
    print(f"Python: {py}")
    print(f"Log   : {log_path}")
    start = datetime.datetime.now()
    with open(log_path, "w", encoding="utf-8") as log:
        for s in scripts:
            code = run_script(py, s, env, log)
            if code != 0:
                print(f"\n{s} stopped with an error (exit code {code}) - see the message "
                      f"above or {log_path}.")
                sys.exit(code)
    mins = (datetime.datetime.now() - start).total_seconds() / 60
    print(f"\nAll done in {mins:.0f} min. Results: the .csv files in this folder and the "
          f"plots folder. Full output saved in {log_path}")


if __name__ == "__main__":
    main()

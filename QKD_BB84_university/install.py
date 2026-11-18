"""
One-off setup: creates a virtual environment and installs everything the project needs,
then runs check_setup.py. Works on Windows, Linux and macOS, no admin rights needed.

    py install.py            (Windows)
    python3 install.py       (Linux / macOS)

Options:
    --venv PATH     where to put the virtual environment
                    (default: Windows %LOCALAPPDATA%\\qkd-bb84-venv, else ~/.venvs/qkd-bb84)
    --no-venv       install into the Python running this script instead (pip --user)
    --gpu           install the normal PyTorch (with GPU support) instead of the CPU-only one
    --proxy URL     use a web proxy for downloads, e.g. http://proxy.uni.ac.uk:8080

Lab PCs that wipe the local disk at logout delete the virtual environment too: just run
install.py again at the start of the session (your code and results are untouched).
"""
import argparse
import os
import platform
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))


def default_venv():
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "qkd-bb84-venv")
    return os.path.join(os.path.expanduser("~"), ".venvs", "qkd-bb84")


def venv_python(path):
    if platform.system() == "Windows":
        return os.path.join(path, "Scripts", "python.exe")
    return os.path.join(path, "bin", "python")


def run(cmd, check=True):
    print("\n> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        print("\nERROR: the command above failed. Read the message above; the "
              "'Troubleshooting' section of README.md covers the usual causes "
              "(no internet / proxy, disk quota, blocked downloads).")
        sys.exit(r.returncode)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=os.environ.get("QKD_VENV") or default_venv())
    ap.add_argument("--no-venv", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--proxy", default=None)
    args = ap.parse_args()

    print("=" * 72)
    print("QKD BB84 project - install")
    print("=" * 72)
    print(f"Project folder : {HERE}")
    print(f"Python         : {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is needed (3.12 recommended). See README.md step 1.")
        sys.exit(1)

    if args.no_venv:
        py = sys.executable
        user = [] if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ["--user"]
        print("Installing into this Python (no virtual environment)")
    else:
        py = venv_python(args.venv)
        user = []
        print(f"Virtual env    : {args.venv}")
        if not os.path.exists(py):
            print("Creating the virtual environment ...", flush=True)
            try:
                os.makedirs(os.path.dirname(args.venv), exist_ok=True)
                venv.create(args.venv, with_pip=True)
            except Exception as e:  # noqa: BLE001
                print(f"\nERROR: could not create the virtual environment: {e}")
                print("Try another location:  py install.py --venv C:\\Temp\\qkd-bb84-venv")
                print("On Ubuntu/Debian you may need:  sudo apt install python3-venv")
                sys.exit(1)

    pip = [py, "-m", "pip", "install"] + user
    if args.proxy:
        pip += ["--proxy", args.proxy]

    print("\nInstalling libraries (5-10 minutes the first time) ...", flush=True)
    run([py, "-m", "pip", "install", "--upgrade", "pip"] + user +
        (["--proxy", args.proxy] if args.proxy else []), check=False)
    if not args.gpu and platform.system() in ("Windows", "Linux"):
        # CPU-only PyTorch is much smaller; if this is blocked, the normal one is
        # installed from requirements.txt on the next line instead.
        run(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=False)
    run(pip + ["-r", os.path.join(HERE, "requirements.txt")])

    print("\nChecking everything works ...", flush=True)
    code = run([py, os.path.join(HERE, "check_setup.py")], check=False)
    print()
    if code == 0:
        print("Setup finished. Next:  py run.py quick   (then  py run.py  for the full run)")
    else:
        print("Setup finished, but check_setup.py found problems - see the [FAIL] lines.")
    if not args.no_venv and args.venv != default_venv():
        print(f"You used a custom location: pass  --venv {args.venv}  to run.py as well.")


if __name__ == "__main__":
    main()

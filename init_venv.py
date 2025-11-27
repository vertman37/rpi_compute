#created by the GPT. linux / windows venv installer using system python.

import os
import sys
import subprocess
import venv
import platform
from pathlib import Path

REQ = "requirements.txt"
VENV_DIR = ".venv"


#to just run the code. (not command python)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # 1) check requirements.txt
    if not Path(REQ).exists():
        print(f"{REQ} not found.")
        sys.exit(1)

    # 2) create venv if not exists
    if not Path(VENV_DIR).exists():
        print("Creating venv...")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)
    else:
        print("venv exists, skipping creation.")

    # 3) Select Python inside venv
    if platform.system() == "Windows":
        py = str(Path(VENV_DIR) / "Scripts" / "python.exe")
        pip = str(Path(VENV_DIR) / "Scripts" / "pip.exe")
    else:
        py = str(Path(VENV_DIR) / "bin" / "python3")
        pip = str(Path(VENV_DIR) / "bin" / "pip")

    # 4) upgrade pip
    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    # 5) install requirements
    run([py, "-m", "pip", "install", "-r", REQ])

    print("DONE.")

if __name__ == "__main__":
    main()

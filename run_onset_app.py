from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None, env=None):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)

def main():
    here = Path(__file__).resolve().parent

    venv_dir = here / ".venv"
    req_file = here / "requirements.txt"
    app_file = here / "onset_app.py"

    if not req_file.exists():
        print(f"ERROR: requirements.txt not found at {req_file}")
        return 1

    if not app_file.exists():
        print(f"ERROR: onset_app.py not found at {app_file}")
        return 1

    py = sys.executable

    if not venv_dir.exists():
        print(f"Creating virtual environment at: {venv_dir}")
        run([py, "-m", "venv", str(venv_dir)])

    if sys.platform.startswith("win"):
        vpy = venv_dir / "Scripts" / "python.exe"
    else:
        vpy = venv_dir / "bin" / "python"

    if not vpy.exists():
        print("ERROR: venv python not found. Delete .venv and rerun.")
        return 1    
        
    run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(vpy), "-m", "pip", "install", "-r", str(req_file)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print("\nLaunching app at: http://localhost:8501\n")
    run([str(vpy), "-m", "streamlit", "run", str(app_file)], cwd=here, env=env)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

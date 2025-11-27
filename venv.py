#python venvrun.py cpb
import os
import sys
import subprocess

venv_dir = ".venv"

#to just run the code. (not command python)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


if sys.platform == "win32":
    py_exe = os.path.join(venv_dir, "Scripts", "python.exe")
else:
    py_exe = os.path.join(venv_dir, "bin", "python3")

if not os.path.exists(py_exe):
    print(f"{py_exe} not found. Did you run setup_venv.py?")
    # sys.exit(1)
    input('press anykey to exit')

script = "main.py"
args = sys.argv[1:]

try:
    py_code = args[0]
except IndexError:
    py_code = 'main'

if not py_code.endswith('.py'):
    py_code = py_code + '.py'

# .venv Python runs the code
subprocess.run([py_exe, py_code])

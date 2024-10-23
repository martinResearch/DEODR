python.exe -m venv python_venv
call %~dp0/python_venv/Scripts/activate.bat
pip install uv
uv pip sync requirements.txt
uv pip install -e .

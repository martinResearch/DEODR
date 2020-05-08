pip  install virtualenv
python -m venv python_venv
.\python_venv\scripts\python.exe -m pip install --upgrade pip
.\python_venv\scripts\pip.exe install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
.\python_venv\scripts\pip.exe install -r requirements.txt
.\python_venv\scripts\pip.exe install -e .

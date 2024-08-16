echo [$(date)] ="START"
echo [$(date)] ="Creating Conda environment with python 3.8"
conda create --prefix ./env python=3.8 -y

echo [$(date)] ="Activating virtual environment"
source activate ./env

echo [$(date)] ="Installing dev requirements.txt"
pip install -r requirements_dev.txt

echo [$(date)] ="END"




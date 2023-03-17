# !/bin/bash
# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2023/03/17


echo 'create an environment {teamname}_task{A,B,C}_venv'
python3 -m venv NUS-IDS_taskA_venv

echo 'activate your environment'
source NUS-IDS_taskA_venv/bin/activate

echo 'install all your requirements'
pip install -r requirements.txt

echo 'then close your environment'
deactivate
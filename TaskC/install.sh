echo 'create an environment {teamname}_task{A,B,C}_venv'
python3 -m venv NUS-IDS_taskC_venv

echo 'activate your environment'
source NUS-IDS_taskC_venv/bin/activate

echo 'install all your requirements'
pip install -r taskc_requirements.txt

echo 'then close your environment'
deactivate
eval "$(conda shell.bash hook)"

conda env create -f ./environment.yml
conda activate $(grep '^name:' ./environment.yml | awk '{print $2}') 
pip install poetry
poetry install --all-extras
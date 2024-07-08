conda env create -f environment.yml
conda activate hexgin
poetry install
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
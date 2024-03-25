# DTG

# VIDEO:
[![IMAGE ALT TEXT HERE](https://github.com/jingGM/DTG/front.png)](https://youtu.be/1-YZwSma5Z4)

# Pre-Print Paper
[DTG : Diffusion-based Trajectory Generation for Mapless Global Navigation](https://arxiv.org/abs/2403.09900)


# Install Environment
```
conda create -n hn python=3.10
conda activate hn
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install -r requirements.txt
```

# Run
```
python main.py 
```
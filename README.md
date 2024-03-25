# DTG

# VIDEO:

<iframe width="560" height="315" src="https://www.youtube.com/embed/1-YZwSma5Z4?si=nkXpd0Fppmzv9fEv" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

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
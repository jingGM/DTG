# DTG

# [VIDEO:](https://www.youtube.com/watch?v=1-YZwSma5Z4)
[![IMAGE ALT TEXT HERE](https://github.com/jingGM/DTG/blob/main/front.png)](https://www.youtube.com/watch?v=1-YZwSma5Z4)

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

Download sample dataset in the root folder:
https://drive.google.com/drive/folders/1YClCBSCUc3_Zy3WIQfAE6_kIQ0xTOe0I?usp=sharing

# Run
- generator_type: 0: diffusion model; 1: cvae
- diffusion_model: 0: crnn; 1:unet
- crnn_type: 0: gru; 1:lstm
```
python main.py --wandb_api=YOUR_WANDB_API --generator_type=0 --diffusion_model=0 --crnn_type=0
```
# DualNeRF

> This is the MEng thesis work of Tianhao (Walter) Wu
> 
> Please note that the code is originally developed based on NeRF's code repository at https://github.com/bmild/nerf
>
> A large portion of their code is reused.

To setup the conda environment:
```
conda env create -f environment.yml
conda activate dual_nerf
```

To begin the training process, first download the ShapeNet car dataset rendered by SRN from https://drive.google.com/file/d/19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU/view?usp=sharing, unzip it into a proper directory and change the corresponding `datadir` line in `configs\config_car_full_gl_srn_data.txt`, then run:
```
python run_nerf_r.py --config config_car_full_gl_srn_data.txt
```

To generate renderings on test dataset using trained model, set `render_only` to `True` in the config file and re-run the above command.
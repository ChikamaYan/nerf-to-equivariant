# DualNeRF

> This is the MEng thesis work of Tianhao (Walter) Wu
> 
> Please note that the code is originally developed based on NeRF's code repository at https://github.com/bmild/nerf
>
> A large portion of their code is therefore re-used.
> 
> A rough reference for re-used code can be found at the beginning for each file.
>
> For a detailed and exact reference on code re-write and re-use, please refer to the commit history, as this code repository was forked directly from the NeRF repository.

# Quick Start

To setup the conda environment:
```
conda env create -f environment.yml
conda activate dual_nerf
```

To begin the training process, first download the ShapeNet car dataset rendered by SRN from https://drive.google.com/file/d/19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU/view?usp=sharing, unzip it into a suitable location and change the corresponding `datadir` line in `configs/config_car_full_gl_srn_data.txt`, then run:
```
python run_nerf_r.py --config configs/config_car_full_gl_srn_data.txt
```

To generate renderings on test dataset using trained model, set `render_only` to `True` in the config file and re-run the above command.

# File Structure

`configs/`: list of configuration files. A document for each config parameter can be found at `utils/config_parser.py`

`model/`: definition of positional embedder and all deep learning models

`utils/`: helper functions including data loader, config parser and rendering helpers.

`render_real.ipynb`: notebook containing real-life dataset rendering experiment.

`run_nerf_r.py`: main training and evaluation functions

`run_nerf.py`: training and evaluation functions for NeRF baseline


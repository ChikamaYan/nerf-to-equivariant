# DualNeRF

## TL;DR quickstart

To setup a conda environment, download example training data, begin the training process, and launch Tensorboard:
```
conda env create -f environment.yml
conda activate nerf
python run_nerf.py --config config_fern.txt
tensorboard --logdir=logs/summaries --port=6006
```
If everything works without errors, you can now go to `localhost:6006` in your browser and watch the "Fern" scene train.

## Setup

Python 3 dependencies:

* Tensorflow 1.15
* matplotlib
* numpy
* imageio
*  configargparse

We provide a conda environment setup file including all of the above dependencies. Create the conda environment `nerf` by running:
```
conda env create -f environment.yml
```

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.

## What is a NeRF?

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.


## Running code

Here we show how to run our code on two example scenes. You can download the rest of the synthetic and real data used in the paper [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

### Optimizing a NeRF

Run
```
bash download_example_data.sh
```
to get the our synthetic Lego dataset and the LLFF Fern dataset.

To optimize a low-res Fern NeRF:
```
python run_nerf.py --config config_fern.txt
```
After 200k iterations (about 15 hours), you should get a video like this at `logs/fern_test/fern_test_spiral_200000_rgb.mp4`:

![ferngif](https://people.eecs.berkeley.edu/~bmild/nerf/fern_200k_256w.gif)

To optimize a low-res Lego NeRF:
```
python run_nerf.py --config config_lego.txt
```
After 200k iterations, you should get a video like this:

![legogif](https://people.eecs.berkeley.edu/~bmild/nerf/lego_200k_256w.gif)

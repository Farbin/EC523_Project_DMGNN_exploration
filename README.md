# EC523 Project: Exploration and Reproduction of DMGNN  
### Dynamic Multiscale Graph Neural Networks for 3D Human Motion Prediction

This repository contains our reproduction and analysis of the **DMGNN** model (*Dynamic Multiscale Graph Neural Network*, CVPR 2020) applied to the **H3.6M** and **CMU Mocap** datasets.  The project explores the scale granularity of DMGNN on human motion prediction performance.

---

## 📌 Project Overview

DMGNN performs 3D human motion prediction, given a short sequence of past skeleton poses, the model predicts future poses. It does so using:

- **Multiscale graph representations** (joint → low-level body part → high-level body part)  
- **Dynamic graph learning** at each layer  
- **Multiscale Graph Convolution Units (MGCUs)**  
- **A Gated Graph Recurrent Unit (G-GRU) decoder**  
- **Three motion streams**: position, velocity, and acceleration  

In this project, we:

1. Retrained the DMGNN model and validated prediction performance  
2. Explored the model architecture with various scale configurations
3. Visualized human motion in 2D and 3D stickman formats
   
---

## Repository Structure

```text
.
├── README.md
├── cmu-long/
│   └── ...
├── cmu-short/
│   ├── main.py
│   ├── net/
│   │   ├── __init__.py
│   │   ├── model.py                     # modified DMGNN model
│   │   └── utils/
│   ├── processor/
│   │   └── ...
│   ├── submit_job.sh
│   └── work_dir/
│       └── prediction/
├── config/
│   ├── CMU/
│   │   ├── long/
│   │   └── short/
|   |      ├── test.yaml
|   |      ├── train.yaml               # Config for Scale 1 only
|   |      ├── train_scale12.yaml       # Config for Scales 1+2
|   |      └── train_scale123.yaml      # Config for Scales 1+2+3
│   └── H36M/
│       ├── long/
│       └── short/
|           ├── test.yaml
|           ├── train.yaml
|           ├── train1.yaml              # Config for Scales 1 only
|           ├── train12.yaml             # Config for Scales 1+2
|           └── train123.yaml            # Config for Scales 1+2+3
├── h36m-long/
│   └── ...
├── h36m-short/
│   ├── main.py
│   ├── net/
│   │   ├── __init__.py
│   │   ├── model.py                     # modified DMGNN model
│   │   └── utils/
│   └── processor/
│       └── ...
├── img/
│   └── ...
└── torchlight/
    └── ...
```

## Setting up the Environment on SCC
Create a Python environment with the following libraries on SCC.
- Python 3.6
- Pytorch 1.0
- pyyaml
- argparse
- numpy
- h5py

Then, run

```cd torchlight, python setup.py install, cd ..```

Done.

## How to Run
First, download the dataset from https://github.com/limaosen0/DMGNN/tree/master/data in the project folder.

### Training
A model can be trained on `cmu-short`/`cmu-long`/`h3m-short`/`h3m-long` separately. To train on either of the datasets, first go to the respective folder. Then run the main.py script with a train configuration file (example: `config/CMU/short/train.yaml`). As an example for the `cmu-short` dataset, run:

``` cd cmu-short ```

``` python3 main.py prediction -c ../config/CMU/short/train.yaml ```

Configure the `train.yaml` file as needed.

### Testing

To test on either of the datasets, first go to the respective folder. Then run the main.py script with a test configuration file (example: `config/CMU/short/test.yaml`). As an example for the `cmu-short` dataset, run:

``` cd cmu-short ```

``` python3 main.py prediction -c ../config/CMU/short/test.yaml ```

Configure the `train.yaml` file as needed.


### Generating Animations

Animations can be generated while testing the `cmu-short` dataset. The procedure is same as above. By default, animations are generated always when you test. To disable or enable this feature, you can use the `save_motion` parameter in the `test` function of `cmu-short/processor/recognition.py`. 


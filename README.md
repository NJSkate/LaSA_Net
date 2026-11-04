LaSA-Net: A Language-Guided Network for Outdoor Generalized 3D Referring Expression Segmentation

NEWS: This work has been accepted by NeurIPS 2026！

Anonymous Authors

### UrbanRefer dataset

The UrbanRefer dataset and the checkpoints is available in the https://drive.google.com/drive/folders/1qtlO2JjENxVdUXC0QqzptftXhgrLfLi0?usp=sharing


## Installation

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n lasanet python=3.8
  conda activate lasanet
  ```

- Clone this repository

  ```
  git clone <anonymous-repository-url>
  ```

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl # please check the versions in the website
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install lasanet and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd lasanet/lib/
  python setup.py develop
  ```
- Compile pointnet++
  ```
  cd pointnet2
  python setup.py install --user
  cd ..
  ```
- Install Pytorch3D. You can try 'pip install pytorch3d'. If it doesn't work, you can install it from source:
  ```
  git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d && pip install -e .
  ```


## Training

The configs are named by dataset:

```bash
configs/lasanet_multi3drefer.yaml
configs/lasanet_scanrefer.yaml
configs/lasanet_urbanrefer.yaml
```

```bash
bash scripts/train_lasanet_multi3drefer.sh   # Multi3DRefer
bash scripts/train_lasanet_scanrefer.sh      # ScanRefer
bash scripts/train_lasanet_urbanrefer.sh     # UrbanRefer
```

## Inference

```bash
bash scripts/test_lasanet_multi3drefer.sh    # Multi3DRefer
bash scripts/test_lasanet_scanrefer.sh       # ScanRefer
bash scripts/test_lasanet_urbanrefer.sh      # UrbanRefer
```


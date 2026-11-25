**LaSA-Net: A Language-Guided Network for Outdoor Generalized 3D Referring Expression Segmentation**
3D Referring Expression Segmentation (3D-RES) aims to segment target objects in point clouds via language. However, extending 3D-RES from indoor scenes to large-scale outdoor environments introduces three major challenges, i.e., large-scale geospatial complexity lacking in indoor benchmarks, sparse-target perception overwhelmed by massive geometric clutter and distractors, and context-induced query dilution, where highly relation-intensive expressions cause globally-initialized queries to be absorbed by salient reference objects. To address these issues, we introduce UrbanRefer, the first outdoor 3D-GRES benchmark with 193 urban road point cloud scenes and 7,840 textual descriptions for complex geospatial reasoning. We further propose LaSA-Net, a Language-guided Semantic Segmentation Network. First, to enhance sparse-target perception, LaSA-Net incorporates pixel-level dense features from DINOv3 as 2D priors and fuses them with 3D features through depth-consistent verification. Then, to alleviate query dilution, we design a semantic-adaptive query generation module that applies text-weighted farthest point sampling to focus on text-relevant regions, and performs multi-granularity sampling to maintain both local grounding and global semantic consistency. Finally, a reliability-aware decoder adaptively suppresses noisy language cues through cross-modal reliability-gated injection, generating precise prompts for accurate mask prediction. Extensive experiments demonstrate that LaSA-Net achieves 52.5%, 55.9%, and 47.9% mIoU on the ScanRefer, Multi3DRefer, and UrbanRefer datasets, surpassing prior state-of-the-art methods by 2.3%, 4.2%, and 2.8% respectively.

<img width="1373" height="479" alt="image" src="https://github.com/user-attachments/assets/feaf6c0d-157e-4f4a-b2c8-f9f9f195dbfc" />

🔥 $\textcolor{red}{\textbf{NEWS:}}$ Our work has been accepted to **NeurIPS 2026**! 🎉

### UrbanRefer dataset

The UrbanRefer dataset is available at the following link:https://drive.google.com/drive/folders/145bEt7QN1sMERWeO43Pg_-II1m3c0K2V?usp=sharing

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


# BFL
This is the official PyTorch implementation of **BFL** (Beyond the Final Layer: A Hierarchical Query Fusion Transformer with Geometry Enhancement for 3D Instance Segmentation).

**Beyond the Final Layer: A Hierarchical Query Fusion Transformer with Geometry Enhancement for 3D Instance Segmentation** [\[Paper\]](TBD)

Jiahao Lu

# Get Started

## Environment

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

The following installation suppose `python=3.8` `pytorch=1.10` and `cuda=11.4`.

- Create a conda virtual environment

  ```
  conda create -n MMImp python=3.8
  conda activate MMImp
  ```

- Install the dependencies

  Install [Pytorch 1.10](https://pytorch.org/)

  ```
  pip install spconv-cu114
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install spformer and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd spformer/lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
MMImp
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

Split and preprocess data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
MMImp
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```
## Training

### ScanNetv2
```
python3 tools/train.py configs/scannet/MMImp_scannet.yaml
```

## Validation
```
python3 tools/test.py configs/scannet/MMImp_scannet.yaml --resume [MODEL_PATH]
```
## Pre-trained Models


| dataset | Model | AP | AP_50% | AP_25% |  Download  |
|---------------|:----:|:----:|:----:|:----:|:-----------:|
| [ScanNetv2] | SPFormers | 56.3 | 73.9 | - | [Model Weight] |
| [ScanNetv2] | MAFT | 58.4 | 75.9 | - | [Model Weight] |
| [ScanNetv2] | SPFormer + Ours | 59.4 | 78.2 | 85.7 | [Model Weight] |
| [ScanNetv2] | MAFT + Ours | 61.4 | 79.3 | 87.0 | [Model Weight] |

| dataset | Model | AP | AP_50% | AP_25% |  Download  |
|---------------|:----:|:----:|:----:|:----:|:-----------:|
| [ScanNet200] | MAFT | 29.2 | 38.2 | 43.3 | [Model Weight] |
| [ScanNet200] | MAFT + Ours | 29.8 | 39.2 | 44.3 | [Model Weight] |
		

# Citation
If you find this project useful, please consider citing:

```
TBD
```

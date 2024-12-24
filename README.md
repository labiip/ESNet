## 代码依赖
 
- Linux 和 macOS 
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ 
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


| MMDetection    |          MMCV          |
|:--------------:|:-------------------------:|
|     master     | mmcv-full>=1.3.17, <1.5.0 |
|     2.23.0     | mmcv-full>=1.3.17, <1.5.0 |
|     2.22.0     | mmcv-full>=1.3.17, <1.5.0 |
|     2.21.0     | mmcv-full>=1.3.17, <1.5.0 |
|     2.20.0     | mmcv-full>=1.3.17, <1.5.0 |
|     2.19.1     | mmcv-full>=1.3.17, <1.5.0 |
|     2.19.0     | mmcv-full>=1.3.17, <1.5.0 |
|     2.18.1     | mmcv-full>=1.3.17, <1.4.0 |
|     2.18.0     | mmcv-full>=1.3.14, <1.4.0 |
|     2.17.0     | mmcv-full>=1.3.14, <1.4.0 |
|     2.16.0     | mmcv-full>=1.3.8, <1.4.0  |
|     2.15.1     | mmcv-full>=1.3.8, <1.4.0  |
|     2.15.0     | mmcv-full>=1.3.8, <1.4.0  |
|     2.14.0     | mmcv-full>=1.3.8, <1.4.0  |
|     2.13.0     | mmcv-full>=1.3.3, <1.4.0  |
|     2.12.0     | mmcv-full>=1.3.3, <1.4.0  |
|     2.11.0     | mmcv-full>=1.2.4, <1.4.0  |
|     2.10.0     | mmcv-full>=1.2.4, <1.4.0  |


## Installation process

### Prepare the environment

1. Create a virtual environment with conda and enter the virtual environment.

   ```shell
   conda create -n mmdet python=3.7 -y
   conda activate mmdet
   ```

2. Install PyTorch and torchvision based on [PyTorch website](https://pytorch.org/), for example:

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

   ```shell
   conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
   ```

### Install MMDetection

1. To install mmcv-full, we recommend using the pre-built package to install:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    You need to replace '{cu_version}' and '{torch_version}' on the command line with the corresponding version. For example, in CUDA 11 and PyTorch 1.7.0 environments, you can install the latest version of MMCV with the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    Refer to [MMCV] (https://mmcv.readthedocs.io/en/latest/#installation) access to different versions of the compatible to different MMCV PyTorch and CUDA version. At the same time, you can also compile MMCV from source by using the following command line:

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  
    cd ..
    ```


    ```shell
    pip install mmcv-full
    ```


    ```
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
    ```

2. Insatll MMDetection：

    ```shell

    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

### Setup script from scratch (sample installation)


```shell
conda create -n mmdet python=3.7 -y
conda activate mmdet

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# Install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html (这里要选择对应当前mmdet的版本的mmcv版本，具体版本查看可查看mmdet/version.py)

# Insatll MMDetection
pip install -r requirements/build.txt
pip install -v -e .
```


## verify

To verify that MMDetection and the required environment are installed correctly, we can run the sample Python code to initialize the detector and reason about a demonstration image:

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
#http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# Initialize the detector
model = init_detector(config_file, checkpoint_file, device=device)
#  Reasoning demo image
inference_detector(model, 'demo/demo.jpg')
```
If MMDetection is installed successfully, the above code runs completely.

## Training code
```python
# The model is trained using the command line
python tools/train.py work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py ```

## Test code
```python
# Test the model using the command line
python tools/test.py work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py work_dirs/esnet/epoch_40.pth```

## Single picture reasoning
```python
# Single-image reasoning on the model using the command line
python demo/image_demo.py demo/test.tif work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py /TEST/xiuyu.li/esnet/epoch_40.pth

```









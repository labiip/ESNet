## 代码依赖

- Linux 和 macOS （Windows 理论上支持）
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ （如果基于 PyTorch 源码安装，也能够支持 CUDA 9.0）
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

MMDetection 和 MMCV 版本兼容性如下所示，需要安装正确的 MMCV 版本以避免安装出现问题。

| MMDetection 版本 |          MMCV 版本          |
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

**注意：**如果已经安装了 mmcv，首先需要使用 `pip uninstall mmcv` 卸载已安装的 mmcv，如果同时安装了 mmcv 和 mmcv-full，将会报 `ModuleNotFoundError` 错误。

## 安装流程

### 准备环境

1. 使用 conda 新建虚拟环境，并进入该虚拟环境；

   ```shell
   conda create -n mmdet python=3.7 -y
   conda activate mmdet
   ```

2. 基于 [PyTorch 官网](https://pytorch.org/)安装 PyTorch 和 torchvision，例如：

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   **注意**：需要确保 CUDA 的编译版本和运行版本匹配。可以在 [PyTorch 官网](https://pytorch.org/)查看预编译包所支持的 CUDA 版本。

   `例 1` 例如在 `/usr/local/cuda` 下安装了 CUDA 10.1， 并想安装 PyTorch 1.5，则需要安装支持 CUDA 10.1 的预构建 PyTorch：

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

   `例 2` 例如在 `/usr/local/cuda` 下安装了 CUDA 9.2， 并想安装 PyTorch 1.3.1，则需要安装支持 CUDA 9.2  的预构建 PyTorch：

   ```shell
   conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
   ```

   如果不是安装预构建的包，而是从源码中构建 PyTorch，则可以使用更多的 CUDA 版本，例如 CUDA 9.0。

### 安装 MMDetection

1. 安装 mmcv-full，我们建议使用预构建包来安装：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    需要把命令行中的 `{cu_version}` 和 `{torch_version}` 替换成对应的版本。例如：在 CUDA 11 和 PyTorch 1.7.0 的环境下，可以使用下面命令安装最新版本的 MMCV：

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    请参考 [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) 获取不同版本的 MMCV 所兼容的的不同的 PyTorch 和 CUDA 版本。同时，也可以通过以下命令行从源码编译 MMCV：

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # 安装好 mmcv-full
    cd ..
    ```

    或者，可以直接使用命令行安装：

    ```shell
    pip install mmcv-full
    ```

    PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv-full 只提供 1.x.0 的编译包。如果你的 PyTorch 版本是 1.x.1，你可以放心地安装在 1.x.0 版本编译的 mmcv-full。

    ```
    # 我们可以忽略 PyTorch 的小版本号
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
    ```

2. 安装 MMDetection：
	cd到README文件路径

    ```shell

    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

### 从零开始设置脚本(示例安装)

假设当前已经成功安装 CUDA 10.1，这里提供了一个完整的基于 conda 安装 MMDetection 的脚本：

```shell
conda create -n mmdet python=3.7 -y
conda activate mmdet

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# 安装最新版本的 mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html (这里要选择对应当前mmdet的版本的mmcv版本，具体版本查看可查看mmdet/version.py)

# 安装 MMDetection
pip install -r requirements/build.txt
pip install -v -e .
```


## 验证

为了验证是否正确安装了 MMDetection 和所需的环境，我们可以运行示例的 Python 代码来初始化检测器并推理一个演示图像：

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
inference_detector(model, 'demo/demo.jpg')
```
如果成功安装 MMDetection，则上面的代码可以完整地运行。

## 训练代码
```python
# 使用命令行对模型进行训练
python tools/train.py work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py ```

## 测试代码
```python
# 使用命令行对模型进行测试
python tools/test.py work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py work_dirs/esnet/epoch_40.pth```

## 单张图片推理
```python
# 使用命令行对模型进行单张图片推理
python demo/image_demo.py demo/test.tif（推理图片的具体路径） work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py /TEST/xiuyu.li/esnet/epoch_40.pth

```

## 若想使用已有环境进行推理测试，可按照下步骤

1. 登录服务器账户 
account: lixiuyu   passwd:123

2. 激活环境
source activate mmdetection

3. 切换目录
cd /home/lixiuyu/mmdetection-master1/

4. 单张图片推理
python demo/image_demo.py demo/test.tif work_dirs/esnet/mask_rcnn_r101_fpn_2x_coco.py /TEST/xiuyu.li/esnet/epoch_40.pth








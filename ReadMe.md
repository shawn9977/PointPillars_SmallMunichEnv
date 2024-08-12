# PointPillars CUDA 环境 Convert to ONNX

Archive: No
Date: July 31, 2024
Status: Not started

## Notes

<aside>
<img src="https://www.notion.so/icons/document_gray.svg" alt="https://www.notion.so/icons/document_gray.svg" width="40px" />

</aside>

<aside>
<img src="https://www.notion.so/icons/document_gray.svg" alt="https://www.notion.so/icons/document_gray.svg" width="40px" />

</aside>

---

本文主要介绍在RTX3070 上搭建 这个项目 (https://github.com/SmallMunich/nutonomy_pointpillars)的过程

**PointPillars Pytorch Model Convert To ONNX, And Using TensorRT to Load this IR(ONNX) for Fast Speeding Inference**

[https://github.com/SmallMunich/nutonomy_pointpillars](https://github.com/SmallMunich/nutonomy_pointpillars)

# 环境搭建

 

## 1 系统环境

安装Ubuntu 20.04.6 LTS 系统

```markdown
 

CPU :                           11th Gen Intel(R) Core(TM) i9-11900T @ 1.50GHz

 
shawn@rt3070:~/nutonomy_pointpillars/second$ uname -r
5.15.0-117-generic

shawn@rt3070:~/nutonomy_pointpillars/second$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.6 LTS
Release:        20.04
Codename:       focal

```

## 2. 安装 NVIDIA 驱动和 CUDA 信息

安装VIDIA 驱动  535.183.01 

```python

(gpu_env) shawn@rt3070:~/nutonomy_pointpillars$ ubuntu-drivers devices

== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00002482sv00001458sd0000408Fbc03sc00i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-470 - distro non-free
driver   : nvidia-driver-535-open - distro non-free
driver   : nvidia-driver-535 - distro non-free
driver   : nvidia-driver-535-server-open - distro non-free recommended
driver   : nvidia-driver-470-server - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin

(gpu_env) shawn@rt3070:~/nutonomy_pointpillars$
(gpu_env) shawn@rt3070:~/nutonomy_pointpillars$

sudo apt install nvidia-driver-535

sudo reboot
nvidia-smi

```

安装CUDA 12.2.0  :[https://developer.nvidia.com/cuda-12-2-0-download-archive](https://developer.nvidia.com/cuda-12-2-0-download-archive)

```python

安装好CUDA 后 需要修改 ~/.bashrc文件 

export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

```bash
shawn@rt3070:~/nutonomy_pointpillars/second$ nvidia-smi
Fri Aug  2 14:05:50 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3070 Ti     Off | 00000000:01:00.0 Off |                  N/A |
| 61%   47C    P2              87W / 310W |   5306MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A       973      G   /usr/lib/xorg/Xorg                          221MiB |
|    0   N/A  N/A      1321      G   /usr/bin/gnome-shell                         93MiB |
|    0   N/A  N/A    185998      C   python                                     4980MiB |
+---------------------------------------------------------------------------------------+

shawn@rt3070:~/nutonomy_pointpillars/second$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0

```

## 3. Python 环境

```bash
shawn@rt3070:~$ python3
Python 3.8.10 (default, Jul 29 2024, 17:02:10)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>

```

## 4.虚拟环境和已安装的包

用系统Python 创建虚拟环境， 并且激活虚拟环境：

```bash
source /home/shawn/test/gpu/gpu_env/bin/activate

```

**列出已安装的包(共参考):**

```bash
(gpu_env) shawn@rt3070:~/test/gpu/gpu_env$ pip list
Package                  Version
------------------------ ----------
filelock                 3.14.0
fire                     0.6.0
fsspec                   2024.5.0
imageio                  2.34.2
importlib_metadata       7.1.0
Jinja2                   3.1.4
lazy_loader              0.4
llvmlite                 0.41.1
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.1
numba                    0.58.0
numpy                    1.24.4
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        8.9.2.26
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.5.40
nvidia-nvtx-cu12         12.1.105
ocnn                     2.2.2
packaging                24.1
pillow                   10.4.0
pip                      24.2
pkg_resources            0.0.0
protobuf                 3.20.0
pybind11                 2.12.0
PyWavelets               1.4.1
scikit-image             0.21.0
scipy                    1.10.1
setuptools               44.0.0
shapely                  2.0.5
six                      1.16.0
sparseconvnet            0.2
sympy                    1.13.1
tensorboardX             2.6.2.2
termcolor                2.4.0
tifffile                 2023.7.10
torch                    2.3.0
torchaudio               2.3.0
torchvision              0.18.0
triton                   2.3.0
typing_extensions        4.12.2
zipp                     3.19.2

```

# 步骤

### **1.**  准备项目 **下载Code**

Add nutonomy_pointpillars/ to your PYTHONPATH.

```
git clone https://github.com/SmallMunich/nutonomy_pointpillars.git

git clone https://github.com/SmallMunich/nutonomy_pointpillars.git

cd /home/shawn/nutonomy_pointpillars

export PYTHONPATH=$PYTHONPATH:/home/shawn/nutonomy_pointpillars/
```

### 2 创建并进入Python虚拟环境

```markdown

- 操作系统：Ubuntu 20.04
- Python版本：3.8
- CUDA版本：12.2

python -m venv gpu_env
source ./gpu_env/bin/activate

```

### 2. 使用pip安装包

```bash
pip install --upgrade pip
pip install shapely pybind11 protobuf scikit-image numba pillow
pip install torch torchvision torchaudio
pip install fire tensorboardX

```

### 3.下载并且 build SparseConvNet  工具包

Finally, install SparseConvNet. This is not required for PointPillars, but the general SECOND code base expects this to be correctly configured. However, I suggest you install the spconv instead of SparseConvNet.

```python
git clone git@github.com:facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash build.sh
# NOTE: if bash build.sh fails, try bash develop.sh instead
```

### 4. 安装系统依赖包

```bash
sudo apt install libboost-all-dev

```

### 5. 设置环境变量

设置前先查看是否有以下文件存在

add following environment variables for numba to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice

```

### 6. 准备原始数据集和目录结构

准备数据目录结构如下：

```
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory

```

下载原始数据集并放置到相应的目录中。下载地址：

- [KITTI 数据集](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
- [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)

Note: PointPillar's protos use         KITTI_DATASET_ROOT=/home/shawn/nutonomy_pointpillars/datasets

修改 ~/.bashrc ， 将 KITTI_DATASET_ROOT=/home/shawn/nutonomy_pointpillars/datasets 写在 ~ /.bashrc 里

### 7. 修改代码

修改代码1

```

编辑 ./nutonomy_pointpillars/second/core/cc/nms/nms_cpu.h 文件，添加以下内容：
M1:

vim ./nutonomy_pointpillars/second/core/cc/nms/nms_cpu.h
#add in nms_cpu.h: 
#include <iostream>

```

修改代码2

```python

M2:

/home/shawn/nutonomy_pointpillars/second/data/kitti_common.py", line 492
  easy_mask = np.ones((len(dims), ), dtype=bool)
    moderate_mask = np.ones((len(dims), ), dtype=bool)
    hard_mask = np.ones((len(dims), ), dtype=bool)
    i = 0
```

修改添加配置修改后 的 ~/.bashrc

```python

export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
export ftp_proxy=http://child-prc.intel.com:913

export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PYTHONPATH=$PYTHONPATH:/home/shawn/nutonomy_pointpillars/
export KITTI_DATASET_ROOT=/home/shawn/nutonomy_pointpillars/datasets/

```

### 7.Process the  dataset

```python
 cd /home/shawn/nutonomy_pointpillars/second
```

**7.2. Create kitti infos:**

```

python create_data.py create_kitti_info_file --data_path=$KITTI_DATASET_ROOT
```

**7.3. Create reduced point cloud:**

```
python create_data.py create_reduced_point_cloud --data_path=$KITTI_DATASET_ROOT
```

**7.4. Create groundtruth-database infos:**

```
python create_data.py create_groundtruth_database --data_path=$KITTI_DATASET_ROOT
```

**7.5. Modify  xyres_16.proto  config file**

```python

vim /home/shawn/nutonomy_pointpillars/second/configs/pointpillars/car/xyres_16.proto

```

 The config file needs to be edited to point to the above datasets:

```
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```

 修改好的文件 如下 供参考  /home/shawn/nutonomy_pointpillars/second/configs/pointpillars/car/xyres_16.proto

```python

(gpu_env) shawn@rt3070:~/nutonomy_pointpillars/second/configs/pointpillars/car$ cat xyres_16.proto
model: {
  second: {
    voxel_generator {
      point_cloud_range : [0, -39.68, -3, 69.12, 39.68, 1]
      voxel_size : [0.16, 0.16, 4]
      max_number_of_points_per_voxel : 100
    }
    num_class: 1
    voxel_feature_extractor: {
      module_class_name: "PillarFeatureNet"
      num_filters: [64]
      with_distance: false
    }
    middle_feature_extractor: {
      module_class_name: "PointPillarsScatter"
    }
    rpn: {
      module_class_name: "RPN"
      layer_nums: [3, 5, 5]
      layer_strides: [2, 2, 2]
      num_filters: [64, 128, 256]
      upsample_strides: [1, 2, 4]
      num_upsample_filters: [128, 128, 128]
      use_groupnorm: false
      num_groups: 32
    }
    loss: {
      classification_loss: {
        weighted_sigmoid_focal: {
          alpha: 0.25
          gamma: 2.0
          anchorwise_output: true
        }
      }
      localization_loss: {
        weighted_smooth_l1: {
          sigma: 3.0
          code_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
      }
      classification_weight: 1.0
      localization_weight: 2.0
    }
    # Outputs
    use_sigmoid_score: true
    encode_background_as_zeros: true
    encode_rad_error_by_sin: true

    use_direction_classifier: true
    direction_loss_weight: 0.2
    use_aux_classifier: false
    # Loss
    pos_class_weight: 1.0
    neg_class_weight: 1.0

    loss_norm_type: NormByNumPositives
    # Postprocess
    post_center_limit_range: [0, -39.68, -5, 69.12, 39.68, 5]
    use_rotate_nms: false
    use_multi_class_nms: false
    nms_pre_max_size: 1000
    nms_post_max_size: 300
    nms_score_threshold: 0.05
    nms_iou_threshold: 0.5

    use_bev: false
    num_point_features: 4
    without_reflectivity: false
    box_coder: {
      ground_box3d_coder: {
        linear_dim: false
        encode_angle_vector: false
      }
    }
    target_assigner: {
      anchor_generators: {
         anchor_generator_stride: {
           sizes: [1.6, 3.9, 1.56] # wlh
           strides: [0.32, 0.32, 0.0] # if generate only 1 z_center, z_stride will be ignored
           offsets: [0.16, -39.52, -1.78] # origin_offset + strides / 2
           rotations: [0, 1.57] # 0, pi/2
           matched_threshold : 0.6
           unmatched_threshold : 0.45
         }
       }

      sample_positive_fraction : -1
      sample_size : 512
      region_similarity_calculator: {
        nearest_iou_similarity: {
        }
      }
    }
  }
}

train_input_reader: {
  #record_file_path: "/data/sets/kitti_second/kitti_train.tfrecord"
  class_names: ["Car"]
  max_num_epochs : 160
  batch_size: 2
  prefetch_size : 25
  max_number_of_voxels: 12000
  shuffle_points: true
  num_workers: 2
  groundtruth_localization_noise_std: [0.25, 0.25, 0.25]
  groundtruth_rotation_uniform_noise: [-0.15707963267, 0.15707963267]
  global_rotation_uniform_noise: [-0.78539816, 0.78539816]
  global_scaling_uniform_noise: [0.95, 1.05]
  global_random_rotation_range_per_object: [0, 0]
  anchor_area_threshold: 1
  remove_points_after_sample: false
  groundtruth_points_drop_percentage: 0.0
  groundtruth_drop_max_keep_points: 15
  database_sampler {
    database_info_path: "/home/shawn/nutonomy_pointpillars/datasets/kitti_dbinfos_train.pkl"
    sample_groups {
      name_to_max_num {
        key: "Car"
        value: 15
      }
    }
    database_prep_steps {
      filter_by_min_num_points {
        min_num_point_pairs {
          key: "Car"
          value: 5
        }
      }
    }
    database_prep_steps {
      filter_by_difficulty {
        removed_difficulties: [-1]
      }
    }
    global_random_rotation_range_per_object: [0, 0]
    rate: 1.0
  }

  remove_unknown_examples: false
  remove_environment: false
  kitti_info_path: "/home/shawn/nutonomy_pointpillars/datasets/kitti_infos_train.pkl"
  kitti_root_path: "/home/shawn/nutonomy_pointpillars/datasets"
}

train_config: {
  optimizer: {
    adam_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate: {
          initial_learning_rate: 0.0002
          decay_steps: 27840 # 1856 steps per epoch * 15 epochs
          decay_factor: 0.8
          staircase: true
        }
      }
      weight_decay: 0.0001
    }
    use_moving_average: false

  }
  inter_op_parallelism_threads: 4
  intra_op_parallelism_threads: 4
  steps: 296960 # 1856 steps per epoch * 160 epochs
  steps_per_eval: 9280 # 1856 steps per epoch * 5 epochs
  save_checkpoints_secs : 1800 # half hour
  save_summary_steps : 10
  enable_mixed_precision: false
  loss_scale_factor : 512.0
  clear_metrics_every_epoch: false
}

eval_input_reader: {
  #record_file_path: "/data/sets/kitti_second/kitti_val.tfrecord"
  class_names: ["Car"]
  batch_size: 2
  max_num_epochs : 160
  prefetch_size : 25
  max_number_of_voxels: 12000
  shuffle_points: false
  num_workers: 3
  anchor_area_threshold: 1
  remove_environment: false
  kitti_info_path: "/home/shawn/nutonomy_pointpillars/datasets/kitti_infos_val.pkl"
  kitti_root_path: "/home/shawn/nutonomy_pointpillars/datasets"
}

```

### 8  **Train**

```

cd /home/shawn/nutonomy_pointpillars/second/
mdir model_out

cd /home/shawn/nutonomy_pointpillars/second/

python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/home/shawn/nutonomy_pointpillars/second/model_out

```

- If you want to train a new model, make sure "/path/to/model_dir" doesn't exist. for example , create a new folder  model_out
- If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
- Training only supports a single GPU.
- Training uses a batchsize=2 which should fit in memory on most standard GPUs.
- On a single 1080Ti, training xyres_16 requires approximately 20 hours for 160 epochs.

### **9.Evaluate**

```
cd /home/shawn/nutonomy_pointpillars/second/

python pytorch/train.py evaluate --config_path=/home/shawn/nutonomy_pointpillars/second/configs/pointpillars/car/xyres_16.proto --model_dir=/home/shawn/nutonomy_pointpillars/second/model_out

```

- Detection result will saved in model_dir/eval_results/step_xxx.
- By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.

Code Change then  evaluate can pass

```python
/home/shawn/nutonomy_pointpillars/second/utils/eval.py

line719:

            print(f"overlap_ranges[:, {i}, {j}]: {overlap_ranges[:, i, j]}")  # 添加这行
            print(f"overlap_ranges[:, {i}, {j}] type: {type(overlap_ranges[:, i, j])}")  # 添加这行
            # min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
            start, stop, num = overlap_ranges[:, i, j]
            num = int(num)  # Convert num to an integer
            min_overlaps[:, i, j] = np.linspace(start, stop, num)
```

### **10 ONNX IR Generate**

**pointpillars pytorch model convert to IR onnx, you should verify some code as follows:**

this python file is : second/pyotrch/models/voxelnet.py

```
        voxel_features = self.voxel_feature_extractor(pillar_x, pillar_y, pillar_z, pillar_i,
                                                      num_points, x_sub_shaped, y_sub_shaped, mask)

        ###################################################################################
        # return voxel_features ### onnx voxel_features export
        # middle_feature_extractor for trim shape
        voxel_features = voxel_features.squeeze()
        voxel_features = voxel_features.permute(1, 0)
```

UNCOMMENT this line: return voxel_features

And Then, you can run convert IR command.

```
cd ~/second.pytorch/second/

(gpu_env) shawn@rt3070:~/nutonomy_pointpillars/second$ python pytorch/train.py onnx_model_generate --config_path=/home/shawn/nutonomy_pointpillars/second/configs/pointpillars/car/xyres_16.proto --model_dir=/home/shawn/nutonomy_pointpillars/second/model_out

```

![Untitled](PointPillars%20CUDA%20%E7%8E%AF%E5%A2%83%20Convert%20to%20ONNX%20531d193e541a4b5e8bacb87defc8222c/Untitled.png)

![Untitled](PointPillars%20CUDA%20%E7%8E%AF%E5%A2%83%20Convert%20to%20ONNX%20531d193e541a4b5e8bacb87defc8222c/Untitled%201.png)

### 11.Convert IR

```python
ovc rpn.onnx --compress_to_fp16=True --output_model=rpn
ovc pfe.onnx --compress_to_fp16=True --output_model=pfe

ovc pfe.onnx --output_model pfe --input pillar_x[1,1,12000,100],pillar_y[1,1,12000,100],pillar_z[1,1,12000,100],pillar_i[1,1,12000,100],num_points_per_pillar[1,12000],x_sub_shaped[1,1,12000,100],y_sub_shaped[1,1,12000,100],mask[1,1,12000,100] --compress_to_fp16=True --verbose

ovc rpn.onnx --output_model rpn --input [1,64,496,432] --compress_to_fp16=True --verbose

每个输入参数的含义如下：

pillar_x[1,1,12000,100]：输入名称为 pillar_x，形状为 [1, 1, 12000, 100]。

含义：pillar_x 表示点云数据中的 x 坐标分量。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
pillar_y[1,1,12000,100]：输入名称为 pillar_y，形状为 [1, 1, 12000, 100]。

含义：pillar_y 表示点云数据中的 y 坐标分量。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
pillar_z[1,1,12000,100]：输入名称为 pillar_z，形状为 [1, 1, 12000, 100]。

含义：pillar_z 表示点云数据中的 z 坐标分量。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
pillar_i[1,1,12000,100]：输入名称为 pillar_i，形状为 [1, 1, 12000, 100]。

含义：pillar_i 表示点云数据中的强度（intensity）分量。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
num_points_per_pillar[1,12000]：输入名称为 num_points_per_pillar，形状为 [1, 12000]。

含义：num_points_per_pillar 表示每个柱中的点的数量。形状 [1, 12000] 表示批量大小为 1，每个点云数据有 12000 个柱。
x_sub_shaped[1,1,12000,100]：输入名称为 x_sub_shaped，形状为 [1, 1, 12000, 100]。

含义：x_sub_shaped 表示经过预处理的 x 坐标分量（例如减去中心点的 x 坐标）。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
y_sub_shaped[1,1,12000,100]：输入名称为 y_sub_shaped，形状为 [1, 1, 12000, 100]。

含义：y_sub_shaped 表示经过预处理的 y 坐标分量（例如减去中心点的 y 坐标）。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
mask[1,1,12000,100]：输入名称为 mask，形状为 [1, 1, 12000, 100]。

含义：mask 表示用于标记有效点的掩码（例如，某些点可能是填充的，需要忽略）。形状 [1, 1, 12000, 100] 表示批量大小为 1，每个点云数据有 12000 个柱，每个柱包含 100 个点。
这些输入参数用于描述点云数据的不同特征，以便在转换过程中正确解析和处理它们。

在这个命令中，输入形状 [1, 64, 496, 432] 表示：

批量大小（Batch size）：1，表示一次处理一个输入样本。
通道数（Channels）：64，表示输入特征图有 64 个通道。
高度（Height）：496，表示输入特征图的高度是 496 像素。
宽度（Width）：432，表示输入特征图的宽度是 432 像素。

```

![Untitled](PointPillars%20CUDA%20%E7%8E%AF%E5%A2%83%20Convert%20to%20ONNX%20531d193e541a4b5e8bacb87defc8222c/Untitled%202.png)

![Untitled](PointPillars%20CUDA%20%E7%8E%AF%E5%A2%83%20Convert%20to%20ONNX%20531d193e541a4b5e8bacb87defc8222c/Untitled%203.png)

在 PointPillars 模型中，PFE（Pillar Feature Encoder）和 RPN（Region Proposal Network）是两个关键组成部分，各自承担不同的功能：

### 1. PFE（Pillar Feature Encoder）

**作用**：将原始的点云数据转换成柱状（pillar）特征向量。

**工作流程**：

- **点云数据处理**：首先，将原始的点云数据根据其空间位置分割成固定大小的网格（pillars）。
- **特征提取**：对于每个 pillar，将其中的点的坐标和其他特征（如反射强度）输入到一个小型的神经网络（通常是一个简单的多层感知机或卷积网络），以提取柱状特征。
- **特征聚合**：将每个 pillar 的特征聚合成一个固定大小的特征向量。
- **特征映射**：这些特征向量被映射到一个伪图像中，形成类似图像的输入，这样就可以利用传统的2D卷积神经网络来进一步处理。

### 2. RPN（Region Proposal Network）

**作用**：在伪图像上生成候选区域，用于后续的目标检测。

**工作流程**：

- **输入特征图**：将由 PFE 生成的伪图像作为输入。
- **卷积操作**：利用一系列的卷积层提取高级特征。
- **区域提议**：使用滑动窗口方法在特征图上生成锚框（anchor boxes），这些锚框是潜在的目标区域。
- **分类和回归**：对于每个锚框，RPN 预测其是否包含目标（分类任务）以及对锚框进行微调（回归任务），以精确定位目标。

### 总结

- **PFE** 负责将原始的点云数据转换成可以被2D卷积神经网络处理的伪图像，提取低级特征。
- **RPN** 接受这些伪图像并生成候选区域，提取高级特征并进行目标的初步检测。

这两个模块共同作用，实现了从原始点云数据到目标检测结果的转换。

### 例子

1. **PFE**：
    - 输入：原始点云数据（例如，每个点有 x, y, z 坐标和反射强度）。
    - 输出：一个 2D 特征图（伪图像），其中每个像素代表一个柱状体（pillar）的特征。
2. **RPN**：
    - 输入：PFE 的输出伪图像。
    - 输出：一系列候选框，每个框包含目标概率和位置调整信息。

这种方法的优势在于，它将高维的点云数据映射到低维的2D特征图上，使得复杂的3D目标检测问题可以借助成熟的2D卷积神经网络技术来解决。

### 12 Compare ONNX model With Pytorch Origin model predicts

- If you want to check this convert model about pfe.onnx and rpn.onnx model, please refer to this py-file: check_onnx_valid.py
- Now, we can compare onnx results with pytorch origin model predicts as follows :
- the pfe.onnx and rpn.onnx predicts file is located: "second/pytorch/onnx_predict_outputs", you can see it carefully.

```
    eval_voxel_features.txt
    eval_voxel_features_onnx.txt
    eval_rpn_features.txt
    eval_rpn_onnx_features.txt
```

- pfe.onnx model compare with origin pfe-layer :
- rpn.onnx model compare with origin rpn-layer :

**Compare ONNX with TensorRT Fast Speed Inference**

- First you needs this environments(onnx_tensorrt envs):

```
      docker pull smallmunich/onnx_tensorrt:latest
```

- If you want to use pfe.onnx and rpn.onnx model for tensorrt inference, please refer to this py-file: tensorrt_onnx_infer.py
- Now, we can compare onnx results with pytorch origin model predicts as follows :
- the pfe.onnx and rpn.onnx predicts file is located: "second/pytorch/onnx_predict_outputs", you can see it carefully.

```
    pfe_rpn_onnx_outputs.txt
    pfe_tensorrt_outputs.txt
    rpn_onnx_outputs.txt
    rpn_tensorrt_outputs.txt
```

- pfe.onnx model compare with tensorrt pfe-layer :
- rpn.onnx model compare with tensorrt rpn-layer :

**Blog Address**

- More Details will be update on my chinese blog:
- export from pytorch to onnx IR blog : [https://blog.csdn.net/Small_Munich/article/details/101559424](https://blog.csdn.net/Small_Munich/article/details/101559424)
- onnx compare blog : [https://blog.csdn.net/Small_Munich/article/details/102073540](https://blog.csdn.net/Small_Munich/article/details/102073540)
- tensorrt compare blog : [https://blog.csdn.net/Small_Munich/article/details/102489147](https://blog.csdn.net/Small_Munich/article/details/102489147)
- wait for update & best wishes.
![image](https://github.com/user-attachments/assets/592f0c4a-3a5a-4234-bb05-654e546e5222)## BF-GAN: Development of an AI-driven Bubbly Flow Image Generation Model Using Bubbly Generative Adversarial Networks<br><sub>Official PyTorch implementation version</sub>

![Display image](./docs/bubbly.gif)
![Display image](./docs/1.png)
![Display image](./docs/2.png)

**BF-GAN**<br>
Wen Zhou1, Shuichiro Miwa1, Yang Liu2, Koji Okamoto1<br>
1Department of Nuclear Engineering and Management, School of Engineering, The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-8654, Japan<br>
2 Mechanical Engineering Department, Virginia Tech, Blacksburg, VA 24061, USA<br>
<br>

Abstract: Acquiring a large number of high-quality two-phase flow images remains a time-consuming and expensive task for developing advanced models in fluid dynamics. To address these issues, a generative AI architecture called bubbly flow generative adversarial networks (BF-GAN) is developed, designed to generate realistic and high-quality bubbly flow images through physically conditioned inputs, namely superficial gas (j_g) and liquid (j_f) velocities.

Firstly, 105 sets of two-phase flow experiments under varying conditions are conducted to collect 278,000 bubbly flow images with physical labels of j_g and j_f as training data. A multi-scale loss function of GAN is then developed, incorporating mismatch loss and feature loss to further enhance the generative performance of BF-GAN. The generative performance of the BF-GAN demonstrates a 39.6% improvement over conventional GAN in all generative AI indicators, establishing for the first time a quantitative benchmark in the domain of bubbly flow. In terms of image correspondence, the luminance, contrast, magnitude, homogeneity, and correlation of BF-GAN generated images are compared with experimental images, showing errors between 2.22% and 24.74%. For two-phase flow parameters, including void fraction, aspect ratio, Sauter mean diameter, and interfacial area concentration, the BF-GAN generated images are compared with experimental data, with errors ranging from 2.3% to 16.6%. The comparative analysis demonstrates that the BF-GAN is capable of generating realistic and high-quality bubbly flow images for any given j_g and j_f within the research scope, and these images align with physical properties. 

BF-GAN offers a generative AI solution for two-phase flow research, substantially lowering the time and cost required to obtain a large number of high-quality image data. The BF-GAN model is available online (https://github.com/zhouzhouwen/BF-GAN).


Keywords: 
Bubbly Flow; Physically Conditioned Deep Learning; Image Generation Model; Generative Adversarial Networks<br>



## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* NVIDIA GPUs with at least 4 GB of memory. We have done all testing and development using RTX A6000 and A6000 ADA GPUs.
* 64-bit Python 3.9 and PyTorch 2.3.1 were utilized in the present study. See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required?  See [Troubleshooting](./docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* In the present study,  nvcc -V: Cuda compilation tools, release 11.6, V11.6.112;  NVIDIA-SMI 550.90.07; Driver Version: 550.90.07;  CUDA Version: 12.4

* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yml`
  - `conda activate BF-GAN`

## Getting started (BF-GAN)

* Pre-trained BF-GAN models are stored as `*.pkl` files at ./BF-GAN-models/ <br>

* More models means more diversity.

* You can download the BF-GAN models at：https://data.mendeley.com/datasets/gtbcrhdnrh/1

```.bash
# Generate bubbly flow images using pre-trained models

python BF-GAN.py --outdir=out --label=0.014,0.260 --seeds=0-33 --network=/home/user/ZHOU-Wen/BF-GAN/BF-GAN-models/network-snapshot-00027-5000.pkl
```

'outdir' will specify the output directory.

'label' refers to the value of jg and jf in m/s. Note that the values of jg and jf are separated by commas and there are no spaces.

'seeds' are used to reproduce the same input. One seed will generate 5 images. List of random seeds (e.g., \'0,1,4-6\')

'network' specifies the model location.

![Display image](./docs/3.png)


## Getting started (Bubble detection)
Pre-trained Bubble detection model is stored as `*.pt` files at ./Bubble_detection_model/weights/.

The current bubble detection model is completely based on the BF-GAN dataset, so their combination is the best. This can be used to detect bubbles and further extract their properties.

The bubble detection model is based on YOLOv5. For the installation of YOLOv5, please see: https://github.com/ultralytics/yolov5 <br>
The detect.py script of YOLOv5 was utilized in the present study.

You can download the Bubble detection models at：https://data.mendeley.com/datasets/9f88nrbz4s/1



```.bash
# Detect bubbly flow images

python detect.py --source "./images" --weight best.pt --imgsz=1024 --save-txt --save-crop --line-thickness=2 --hide-labels --hide-conf

```

![Display image](./docs/4.png)



## Pre-generative bubbly flow images

To facilitate everyone to reduce usage costs, red boundaries were delineated based on the current distribution of the dataset. Subsequently, increments of 5% in j_g and j_f were applied, denoted by blue points, resulting in a total of 2080 j_g and j_f conditions. Each condition was generated by six BF-GAN models, generating a total of 3000 images per condition (500 images per model). Consequently, a dataset comprising 6.24 million bubble flow images was constructed, corresponding to the blue-point j_g and j_f conditions.


You can download the Pre-generative bubbly flow images at：


![Display image](./docs/5.png)
## Citation

```
2024
BF-GAN: Development of an AI-driven Bubbly Flow Image Generation Model Using Bubbly Generative Adversarial Networks
Wen Zhou1, Shuichiro Miwa1, Yang Liu2, Koji Okamoto1
```


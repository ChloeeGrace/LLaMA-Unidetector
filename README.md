<h1 align="center">☀️LLaMA-Unidetector☀️</h1>

<h3 align="center" style="font-family: 'Times New Roman'; font-size: 10px;">
LLaMA-Unidetector: An LLaMA-Based Universal Framework for Open-Vocabulary Object Detection in Remote Sensing Imagery
</h3>

<p align="center">
Jianlin Xie<sup>1</sup>, Guanqun Wang<sup>2</sup><sup>*</sup>, Tong Zhang<sup>1</sup>, Yikang Sun<sup>1</sup>, He chen<sup>1</sup>, Yin Zhuang<sup>1</sup>, Jun Li<sup>3</sup>
</p>

<p align="center">
<sup>1</sup> Beijing Institute of Technology, <sup>2</sup> Peking University, <sup>2</sup> China University of Geosciences
</p>

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">🔥Updates</h2>

- 🗓️**May 5th, 2025**: The LLaMA-Unidetector repo has been further optimized.

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">🎯Overview & Contribution</h2>

![Example Image](img/method.png)

Our main contributions are:
- We introduce a novel open-set benchmark for remote sensing, accompanied by a self-built Vision Question Answering (VQA) remote sensing dataset, TerraVQA, providing a platform for researchers to explore open-vocabulary object detection tasks in remote sensing imagery.
- We propose a class-agnostic detector that eliminates multi-class dependencies while explicitly modeling both geometric and probabilistic aspects of object detection, enabling the localization branch to learn general spatial representations and achieve strong generalization across diverse scenarios and unseen object categories.
- We propose a LLaMA-based multimodal large language model without class vocabulary limitations that offers superior semantic understanding and the flexibility to handle novel concepts beyond the vocabulary.

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">
🧾Getting Started
</h2>

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">
1. Installation
</h3>

LLaMA-Unidetector consists of a two-stage algorithm, namely **object localization** and **foreground recognition**. Therefore, virtual environments were established respectively for the two stages.
- **Object localization** is developed based on python==3.8.19 torch==1.13.0+cu117 and torchvision==0.14.0+cu117. Check more details in requirements_localization.txt. 
- **Foreground recognition** is developed based on python==3.8.19 torch==1.13.1+cu117 and torchvision==0.14.1+cu117. Check more details in requirements_recognition.txt.

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">
i. Clone Project
</h3>

```
git clone https://github.com/ChloeeGrace/LLaMA-Unidetector.git
```

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">ii. Install</h3>

<div style="padding-left: 20px;">
    <h3 style="font-family: 'Times New Roman'; font-size: 15px;">    a. Object localization</h3>
    <pre>
    pip install -r requirements_localization.txt
    </pre>
</div>
    
<div style="padding-left: 20px;">
    <h3 style="font-family: 'Times New Roman'; font-size: 15px;">b. Foreground recognition</h3>
    <pre>
pip install -r requirements_recognition.txt
    </pre>
</div>

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">
iii. Download pretrain backbone weight
</h3>

Download the pre-trained [checkpoint0033_4scale.pth](https://drive.usercontent.google.com/download?id=1AwUn5EebmmLBo7njjW_Ng1q9zDrqkNbB&export=download&authuser=0&confirm=t&uuid=310c932c-5d4d-4d53-93ff-0a1d490371d9&at=ALoNOgmhqR4P-8nW4jU6Qbn-Yu5M:1746691948748) weights, and then modify the corresponding path (directory named ckpt in object-localization).

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">
2. Data Preparation
</h3>

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">
i. class-agnostic detection stage
</h3>

A simple example of a vision-language-answering [**(VQA)**](VQA_dataset/AID30_LLAVA_TUNE.json) dataset. Any detection or recognition dataset can be made into a VQA dataset.

<h3 style="font-family: 'Times New Roman'; font-size: 15px;">
ii. TerraOV-LLM foreground recoginition stage
</h3>

Some simple examples of the foreground-background [**(FB)**](class%20agnostic%20detection%20dataset/00002.txt) dataset. Any remote sensing object detection dataset can be made into a FB dataset.

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">🏋️‍♂️Training</h2>

```
bash training.sh
```

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">🤖Inference</h2>

```
bash Inference.sh
```
<h2 style="font-family: 'Times New Roman'; font-size: 15px;">🔗Citation</h2>

```
@ARTICLE{10976651,
  author={Xie, Jianlin and Wang, Guanqun and Zhang, Tong and Sun, Yikang and Chen, He and Zhuang, Yin and Li, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={LLaMA-Unidetector: A LLaMA-Based Universal Framework for Open-Vocabulary Object Detection in Remote Sensing Imagery}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Object detection;Remote sensing;Training;Detectors;Visualization;Feature extraction;Location awareness;Semantics;Data mining;Optimization;open-vocabulary;remote sensing object detection;decoupled learning},
  doi={10.1109/TGRS.2025.3564332}}
```

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">🔔Notice</h2>

We will release our code and VQA dataset to support future research in remote sensing open-vocabulary object detection.

<h2 style="font-family: 'Times New Roman'; font-size: 15px;">📢Contact</h2>

If you have any questions, suggestions or spot a bug, feel free to get in touch. We would also love to see your contributions. Just open a pull request if you'd like to help out. Thanks so much for your support!


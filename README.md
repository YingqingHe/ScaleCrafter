# ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models


<div align="center">

 <a href=''><img src='https://img.shields.io/badge/ArXiv-2305.18247-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href=''><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
  

_**[Yingqing He*](https://github.com/YingqingHe), [Shaoshu Yang*](), [Haoxin Chen](), [Xiaodong Cun](http://vinthony.github.io/), [Menghan Xia](https://menghanxia.github.io/), <br> 
[Yong Zhang<sup>#](https://yzhang2016.github.io), [Xintao Wang](https://xinntao.github.io/), [Ran He](), [Qifeng Chen<sup>#](https://cqf.io/), and [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ)**_

(* first author, # corresponding author)

</div>



## 📝 Changelog
- __[2023.10.09]__: 🔥 Release paper and source code.
<br>

<!-- ## ⏳ TODO
-  -->


## ⚙️ Setup
```bash
conda create -n scalecrafter python=xxx
conda activate scalecrafter
pip install -r requirements.txt
```

## 💫 Inference
```bash
```

## 🔆 Abstract
<b>TL; DR: 🤗🤗🤗 **ScaleCrafter:** A tuning-free approach that can generate images with resolution of 4096x4096 based on pre-trained diffusion models, which is 16 times higher than the original training resolution.</b>

> In this work, we investigate the capability of generating images from pre-trained diffusion models at much higher resolutions than the training image sizes. In addition, the generated images should have arbitrary image aspect ratios. When generating images directly at a higher resolution, 1024 x 1024, with the pre-trained Stable Diffusion using training images of resolution 512 x 512, we observe persistent problems of object repetition and unreasonable object structures. Existing works for higher-resolution generation, such as attention-based and joint-diffusion approaches, cannot well address these issues. As a new perspective, we examine the structural components of the U-Net in diffusion models and identify the crucial cause as the limited perception field of convolutional kernels. Based on this key observation, we propose a simple yet effective re-dilation that can dynamically adjust the convolutional perception field during inference. We further propose the dispersed convolution and noise-damped classifier-free guidance, which can enable ultra-high-resolution image generation (e.g., 4096 x 4096). Notably, our approach does not require any training or optimization. Extensive experiments demonstrate that our approach can address the repetition issue well and achieve state-of-the-art performance on higher-resolution image synthesis, especially in texture details. Our work also suggests that a pre-trained diffusion model trained on low-resolution images can be directly used for high-resolution visual generation without further tuning, which may provide insights for future research on ultra-high-resolution image and video synthesis.




## 😉 Citation
```bib
TODO
```

## 📭 Contact
If your have any comments or questions, feel free to contact [Yingqing He](yhebm@connect.ust.hk) or [Shaoshu Yang](shaoshuyang2020@outlook.com).


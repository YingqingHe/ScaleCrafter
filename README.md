# ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models


<div align="center">

 <a href=''><img src='https://img.shields.io/badge/ArXiv-2305.18247-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://yingqinghe.github.io/scalecrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/YingqingHe/ScaleCrafter-ptl'><img src='https://img.shields.io/badge/ptl version-code-blue'></a> 
 

_**[Yingqing He*](https://github.com/YingqingHe), [Shaoshu Yang*](), [Haoxin Chen](), [Xiaodong Cun](http://vinthony.github.io/), [Menghan Xia](https://menghanxia.github.io/), <br> 
[Yong Zhang<sup>#](https://yzhang2016.github.io), [Xintao Wang](https://xinntao.github.io/), [Ran He](), [Qifeng Chen<sup>#](https://cqf.io/), and [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ)**_

(* first author, # corresponding author)

</div>

## 🔆 Abstract
<b>TL; DR: 🤗🤗🤗 **ScaleCrafter:** A tuning-free approach that can generate images with resolution of 4096x4096 based on pre-trained diffusion models, which is 16 times higher than the original training resolution.</b>

> In this work, we investigate the capability of generating images from pre-trained diffusion models at much higher resolutions than the training image sizes. In addition, the generated images should have arbitrary image aspect ratios. When generating images directly at a higher resolution, 1024 x 1024, with the pre-trained Stable Diffusion using training images of resolution 512 x 512, we observe persistent problems of object repetition and unreasonable object structures. Existing works for higher-resolution generation, such as attention-based and joint-diffusion approaches, cannot well address these issues. As a new perspective, we examine the structural components of the U-Net in diffusion models and identify the crucial cause as the limited perception field of convolutional kernels. Based on this key observation, we propose a simple yet effective re-dilation that can dynamically adjust the convolutional perception field during inference. We further propose the dispersed convolution and noise-damped classifier-free guidance, which can enable ultra-high-resolution image generation (e.g., 4096 x 4096). Notably, our approach does not require any training or optimization. Extensive experiments demonstrate that our approach can address the repetition issue well and achieve state-of-the-art performance on higher-resolution image synthesis, especially in texture details. Our work also suggests that a pre-trained diffusion model trained on low-resolution images can be directly used for high-resolution visual generation without further tuning, which may provide insights for future research on ultra-high-resolution image and video synthesis.


## 📝 Changelog
- __[2023.10.09]__: 🔥 Release paper and source code.
<br>

<!-- ## ⏳ TODO
-  -->


## ⚙️ Setup
```bash
conda create -n scalecrafter python=3.8
conda activate scalecrafter
pip install -r requirements.txt
```

---

## 💫 Inference

### Text-to-image higher-resolution generation with diffusers script
### stable-diffusion xl v1.0 base 
```bash
# 2048x2048 (4x) generation
python3 text2image_xl.py \
--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
--validation_prompt "a professional photograph of an astronaut riding a horse" \
--seed 23 \
--config ./configs/sdxl_2048x2048.yaml \
--logging_dir ${your-logging-dir}
```

To generate in other resolutions, change the value of the parameter `--config` to:
+ 2048x2048: `./configs/sdxl_2048x2048.yaml`
+ 2560x2560: `./configs/sdxl_2560x2560.yaml`
+ 4096x2048: `./configs/sdxl_4096x2048.yaml`
+ 4096x4096: `./configs/sdxl_4096x4096.yaml`

Generated images will be saved to the directory set by `${your-logging-dir}`. You can use your customized prompts by setting `--validation_prompt` to a prompt string or a path to your custom `.txt` file. Make sure different prompts are in different lines if you are using a `.txt` prompt file.

`--pretrained_model_name_or_path` specifies the pretrained model to be used. You can provide a huggingface repo name (it will download the model from huggingface first), or a local directory where you save the model checkpoint.

You can create your custom generation resolution setting by creating a `.yaml` configuration file and specifying the layer to use our method and its dilation scale. Please see `./assets/dilate_setttings/sdxl_2048x2048_dilate.txt` as an example.

### stable-diffusion v1.5 and stable-diffusion v2.1 

```bash
# sd v1.5 1024x1024 (4x) generation
python3 text2image.py \
--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
--validation_prompt "a professional photograph of an astronaut riding a horse" \
--seed 23 \
--config ./configs/sd1.5_1024x1024.yaml \
--logging_dir ${your-logging-dir}

# sd v2.1 1024x1024 (4x) generation
python3 text2image.py \
--pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
--validation_prompt "a professional photograph of an astronaut riding a horse" \
--seed 23 \
--config ./configs/sd2.1_1024x1024.yaml \
--logging_dir ${your-logging-dir}
```
To generate in other resolutions please use the following config files:
+ 1024x1024: `./configs/sd1.5_1024x1024.yaml` `./configs/sd2.1_1024x1024.yaml`
+ 1280x1280: `./configs/sd1.5_1280x1280.yaml` `./configs/sd2.1_1280x1280.yaml`
+ 2048x1024: `./configs/sd1.5_2048x1024.yaml` `./configs/sd2.1_2048x1024.yaml`
+ 2048x2048: `./configs/sd1.5_2048x2048.yaml` `./configs/sd2.1_2048x2048.yaml`

Please see the instructions above to use your customized text prompt.

---

## 😉 Citation
```bib
TODO
```

## 📭 Contact
If your have any comments or questions, feel free to contact [Yingqing He](yhebm@connect.ust.hk) or [Shaoshu Yang](shaoshuyang2020@outlook.com).


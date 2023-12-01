# ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models


<div align="center">

 <a href='https://arxiv.org/abs/2310.07702'><img src='https://img.shields.io/badge/ArXiv-2310.07702-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://yingqinghe.github.io/scalecrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/YingqingHe/ScaleCrafter-ptl'><img src='https://img.shields.io/badge/lightning version-code-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![Replicate](https://replicate.com/cjwbw/scalecrafter/badge)](https://replicate.com/cjwbw/scalecrafter) 
 

_**[Yingqing He*](https://github.com/YingqingHe), [Shaoshu Yang*](https://github.com/ssyang1999), [Haoxin Chen](https://github.com/scutpaul), [Xiaodong Cun](http://vinthony.github.io/), [Menghan Xia](https://menghanxia.github.io/), <br> 
[Yong Zhang<sup>#](https://yzhang2016.github.io), [Xintao Wang](https://xinntao.github.io/), [Ran He](https://rhe-web.github.io/), [Qifeng Chen<sup>#](https://cqf.io/), and [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ)**_

(* first author, # corresponding author)

<img src=assets/pics/video.gif>
Input: "A beautiful girl on a boat"; Resolution: 2048 x 1152.
<br><br>
<img src=assets/pics/img.jpg>
Input: "Miniature house with plants in the potted area, hyper realism, dramatic ambient lighting, high detail"; Resolution: 4096 x 4096.
<br><br>
<img src=assets/pics/anyres.jpg>
Arbitrary higher-resolution generation based on SD 2.1.
<br><br>
</div>

## 🤗 TL; DR
ScaleCrafter is capable of generating images with a resolution of <i>4096 x 4096</i> and videos with a resolution of <i>2048 x 1152</i> based on pre-trained diffusion models on a lower resolution. Notably, our approach needs <i>no extra training/optimization</i>.

## :notes: Notes
- Welcome everyone to collaborate on the code repository, improve methods, and do more downstream tasks. Please check the [CONTRIBUTING.md](https://github.com/YingqingHe/ScaleCrafter/blob/main/CONTRIBUTING.md)
- If you have any questions or comments, we are open for discussion.

## 🔆 Abstract
> In this work, we investigate the capability of generating images from pre-trained diffusion models at much higher resolutions than the training image sizes. In addition, the generated images should have arbitrary image aspect ratios. When generating images directly at a higher resolution, 1024 x 1024, with the pre-trained Stable Diffusion using training images of resolution 512 x 512, we observe persistent problems of object repetition and unreasonable object structures. Existing works for higher-resolution generation, such as attention-based and joint-diffusion approaches, cannot well address these issues. As a new perspective, we examine the structural components of the U-Net in diffusion models and identify the crucial cause as the limited perception field of convolutional kernels. Based on this key observation, we propose a simple yet effective re-dilation that can dynamically adjust the convolutional perception field during inference. We further propose the dispersed convolution and noise-damped classifier-free guidance, which can enable ultra-high-resolution image generation (e.g., 4096 x 4096). Notably, our approach does not require any training or optimization. Extensive experiments demonstrate that our approach can address the repetition issue well and achieve state-of-the-art performance on higher-resolution image synthesis, especially in texture details. Our work also suggests that a pre-trained diffusion model trained on low-resolution images can be directly used for high-resolution visual generation without further tuning, which may provide insights for future research on ultra-high-resolution image and video synthesis.


## 📝 Changelog
- __[2023.10.12]__: 🔥 Release paper.
- __[2023.10.12]__: 🔥 Release source code of both diffuser version and lightning version.
- __[2023.10.16]__: Integrate [FreeU](https://github.com/ChenyangSi/FreeU) as the default mode to further improve our higher-res generation quality. (If you want disable this function, add `--disable_freeu`).
<br>

## ⏳ TODO
- [ ] Hugging Face Gradio demo
- [ ] ScaleCrafter with more controls (e.g., ControlNet/T2I Adapter)

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

****

## 💫 Solve convolution dispersion transform 
We implement MATLAB functions to achieve convolution dispersion. To use the functions, change your MATLAB working directory to `/disperse`. Solve the convlution dispersion transform with
```python
# Small kernel 3, large kernel 5, input feature size 3, perceptual field enlarge scale 2
# Loss weighting 0.05, verbose (deliver visualization) true 
R = kernel_disperse(3, 5, 3, 2, 0.05, true)
```
Then one can save the transform by right-clicking `R` in the workspace window and save this parameter in `.mat` format. We recommend using input feature size to match the size of small kernel, since it can speed up the computation. 
Empirically, this performs well for all convolution kernels in the UNet. 
One can also compute a specific dispersion transform for every input feature size in the diffusion model UNet.

---
## 🤗 Crafter Family
🔥 [LongerCrafter](https://github.com/arthur-qiu/LongerCrafter): Tuning-free method for longer high-quality video generation.  
🔥 [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter): Framework for high-quality video generation.  
🔥 [TaleCrafter](https://github.com/AILab-CVC/TaleCrafter): An interactive story visualization tool that supports multiple characters.  

## 😉 Citation
```bib
@article{he2023scalecrafter,
      title={ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models}, 
      author={Yingqing He and Shaoshu Yang and Haoxin Chen and Xiaodong Cun and Menghan Xia and Yong Zhang and Xintao Wang and Ran He and Qifeng Chen and Ying Shan},
      year={2023},
      eprint={2310.07702},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 📭 Contact
If your have any comments or questions, feel free to contact [Yingqing He](yhebm@connect.ust.hk) or [Shaoshu Yang](shaoshuyang2020@outlook.com).


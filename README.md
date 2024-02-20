# AnimateDiff MotionDirector (DiffDirector)

*Training video*           |  *Output video*
:-------------------------:|:-------------------------:
![myimage](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/b4e2c1d5-d33b-47dc-b106-9836737d3bd2) | ![myimage](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/b61b7919-2c9b-4556-aff9-4c15bb60ebf3)  | 
![bat_run](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/bd40ce25-9892-49f9-9fd4-785e492c04b5) *Output video* | ![rabbit_run](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/f24d05bb-67fa-48d4-a103-610e8de2ffe6) 


*a highly realistic video of batman running in a mystic forest, depth of field, epic lights, high quality, trending on artstation*

This repository is an implementation of [MotionDirector](https://github.com/showlab/MotionDirector) for [AnimateDiff](https://arxiv.org/abs/2307.04725).

AnimateDiff is a plug-and-play module turning most community models into animation generators, without the need of additional training.

MotionDirector is a method to train the motions of videos, and use those motions to drive your animations.

This repository contains the necessary code for training, with intended usage in other applications (such as ComfyUI).
Code is developed user first, and modularized for developers and researchers to build on top of.
Most code has been stripped from the original repository, and runs without distributed mode for ease of use and compatibility. 
If you wish to add it back, you can do so by referencing the original code.

Only `v3` modules have been tested for [Stable Diffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). 
At the moment, there are no plans for SDXL as it's still in early stages, but is very well possible. PRs welcomed! 

### Setup repository and conda environment

```
git clone https://github.com/ExponentialML/AnimateDiff-MotionDirector
cd AnimateDiff-MotionDirector

conda env create -f environment.yaml
conda activate animatediff-motiondirector

pip install -r requirements.txt
```

### Download Stable Diffusion V1.5

```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/
```

### Download V3 Motion Module
```
git lfs install
https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt
```

## Training Instuctions

Open `./configs/training/motion_director/my_video.yaml`.

This is a config that extracts the complexity from the original config, making multiple runs and provides better ease of use.

To setup training, open the `my_video.yaml` in a text editor. There, every line necessary for training will have a comment with instructions necessary to start.

To run, simply execute the following code.

`python train.py --config ./configs/training/motion_director/my_video.yaml`

By default, results will be saved in `outputs`. 

Expected training time on the `"preferred"` quality setting should take roughly 10-15 minutes to converge on roughly 14GB of VRAM.
You can change `max_train_steps` in `.configs/training/motion_director/training.yaml` if you wish to train longer.

### Recommendations
My recommended training workflow for MotionDirector is to choose a single video of the motion you wish to train.

Also, test with and without the domain adapter LoRA for training. While training without the adapter gives results faster, I perceive better training stability in using it rather than not. This also (seemingly) helps with combining with other spatial LoRAs if you chose the latter. 

If you do wish to train multi videos, choose 3-5 videos with similar motions for the best results. Long training runs with large, customized datasets should work theoritically, but this has not been tested. For now, it may be better to train similarly to standard LoRA training (one character, one style, etc.)

More often than not, it seems to work more consistently than multiple videos, and trains faster (500 - 700 steps with AdamW).

Training multiple videos of different motions / subjects and prompts has not been tested thoroughly, so your mileage may vary.

From my tests, a general rule of thumb is 300 steps per video at default settings with AdamW, and about half that with the Lion optimizer.

## Advanced Training Instructions

Intended for developers, researchers, or those who are familiar with finetuning can edit the following config.

`.configs/training/motion_director/training.yaml`

There, you can tweak the learning rate, video datasets, and so on as you see fit.
For more information on dataset handling in code, refer to `./animatediff/utils/dataset.py`.

## Inference

After training, the LoRAs are intended to be used with the ComfyUI Extension [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved).

Simply follow the instructions in the aforementioned repository, and use the AnimateDiffLoader. Any AnimateDiff workflow will work with these LoRAs (including RGB / SparseCtrl).

Two LoRA's will be saved after training in the folders `spatial` & `temporal`, inside of the `output` training folder.

The spatial LoRAs are akin to the ***image*** data, similar to image finetuning. Using these will make your LoRA's look like the subject/s you've trained.

The temporal LoRAs are the ***time*** data (or motion data). Using these will derive the motion of a subject (you can swap a car for an elephant running for instance).

Workflows will be available in the future, but a good place to start is to use IPAdapter in ComfyUI alongside with AnimateDiff using the trained LoRA's from this repository.

### Compatibility

The temporal LoRAs are saved in the same format as MotionLoRAs, so any repository that supports MotionLoRA should be used for them, and will not work otherwise.

To use the spatial LoRAs, load them like any other LoRA. They are saved in CompVis format, ***not*** Diffusers format as the intended usage is for community based repositories.

Alternatively, you can run the inference code with the original AnimateDiff repository, or others that support it.

### More Examples
*Training video*           |  *Output video*
:-------------------------:|:-------------------------:
![car_input](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/75318840-1791-4dee-97be-3f100c27c0b6) | ![boat_drift](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/3c1064d0-4dcc-4792-b77d-5e66443fbe3b)
-> | ![car_drift](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/6d873fa9-fafa-4701-af1c-750b20b1a664)
-> | ![bunny](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/f0ed706d-1cb8-46bf-ad20-be782fb022fe)
-> | ![mk](https://github.com/ExponentialML/AnimateDiff-MotionDirector/assets/59846140/5815c1c0-5d64-4de2-9f09-738bc4ccb840)

## BibTeX

```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Wang, Yaohui and Qiao, Yu and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}
@article{zhao2023motiondirector,
  title={MotionDirector: Motion Customization of Text-to-Video Diffusion Models},
  author={Zhao, Rui and Gu, Yuchao and Wu, Jay Zhangjie and Zhang, David Junhao and Liu, Jiawei and Wu, Weijia and Keppo, Jussi and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2310.08465},
  year={2023}
}
```

## Disclaimer
This project is released for academic use and creative usage. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

## Acknowledgements
Codebase built upon:
- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [Tune-a-Video](https://github.com/showlab/Tune-A-Video).
- [MotionDirector](https://github.com/showlab/MotionDirector)
- [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning)
- [lora](https://github.com/cloneofsimo/lora)

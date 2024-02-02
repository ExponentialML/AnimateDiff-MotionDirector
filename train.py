import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torchvision
import torch.nn.functional as F

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, load_diffusers_lora, load_weights
from animatediff.utils.lora_handler import LoraHandler
from animatediff.utils.lora import extract_lora_child_module
from animatediff.utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset, VID_TYPES
from animatediff.utils.configs import get_simple_config
from lion_pytorch import Lion

augment_text_list = [
    "a video of",
    "a high quality video of",
    "a good video of",
    "a nice video of",
    "a great video of",
    "a video showing",
    "video of",
    "video clip of",
    "great video of",
    "cool video of",
    "best video of",
    "streamed video of",
    "excellent video of",
    "new video of",
    "new video clip of",
    "high quality video of",
    "a video showing of",
    "a clear video showing",
    "video clip showing",
    "a clear video showing",
    "a nice video showing",
    "a good video showing",
    "video, high quality,"
    "high quality, video, video clip,",
    "nice video, clear quality,",
    "clear quality video of"
]

def create_save_paths(output_dir: str):
    lora_path = f"{output_dir}/lora"

    directories = [
        output_dir,
        f"{output_dir}/samples",
        f"{output_dir}/sanity_check",
        lora_path
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    return lora_path

def get_train_dataset(dataset_types, train_data, tokenizer):
    def process_folder_of_videos(train_datasets: list, video_folder: str):
         for video_file in os.listdir(video_folder):

            is_video = any([video_file.split(".")[-1] in ext for ext in VID_TYPES])

            if is_video:
                train_data["single_video_path"] = f"{video_folder}/{video_file}"
                train_datasets.append(SingleVideoDataset(**train_data, tokenizer=tokenizer))

    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                video_folder = train_data.get("path", "")

                if os.path.exists(video_folder) and dataset == "folder":
                            process_folder_of_videos(
                                train_datasets, 
                                video_folder
                            )
                            continue
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))

    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def get_cached_latent_dir(c_dir):
    from omegaconf import ListConfig

    if isinstance(c_dir, str):
        return os.path.abspath(c_dir) if c_dir is not None else None
    
    if isinstance(c_dir, ListConfig):
        c_dir = OmegaConf.to_object(c_dir)
        return c_dir

    return None

def handle_cache_latents(
        should_cache, 
        output_dir, 
        train_dataloader, 
        train_batch_size, 
        vae, 
        cached_latent_dir=None,
        shuffle=False,
        minimum_required_frames=16,
        sampler=None,
        device='cuda'
    ):

    # Cache latents by storing them in VRAM. 
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: 
        return None
    
    vae_dtype = vae.dtype
    vae.to(device, dtype=torch.float32)

    if hasattr(vae, 'enable_slicing'):
        vae.enable_slicing()
    
    cached_latent_dir = get_cached_latent_dir(cached_latent_dir)

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)
    
        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            frames = batch['pixel_values'].shape[1]

            not_min_frames = frames > 2 and frames < minimum_required_frames
            not_img_train = (frames == 1 and batch['dataset'] != 'image')

            if any([not_min_frames, not_img_train]) and minimum_required_frames != 0:
                print(f"""
                    Batch item at index {i} does not meet required minimum frames.
                    Seeing this error means that some of your video lengths are too short, but training will continue.
                    Minimum Frames: {minimum_required_frames}
                    Batch item frames: Batch index = {i}, Batch Frames = {frames}
                    """
                )
                continue

            save_name = f"cached_{i}"
            full_out_path =  f"{cache_save_dir}/{save_name}.pt"

        
            pixel_values = batch['pixel_values'].to(device, dtype=torch.float32)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            
            for k, v in batch.items(): 
                batch[k] = v[0]

            torch.save(batch, full_out_path)
            
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir
        
    # Convert string to list of strings for processing if we have more than.
    cache_save_dir = (
                [cache_save_dir] if not isinstance(cache_save_dir, list) 
            else 
                cache_save_dir
        )

    cached_dataset_list = []

    for save_dir in cache_save_dir:
        cached_dataset = CachedDataset(cache_dir=save_dir)
        cached_dataset_list.append(cached_dataset)

    if len(cached_dataset_list) > 1:
        print(f"Found {len(cached_dataset_list)} cached datasets. Merging...")
        new_cached_dataset = torch.utils.data.ConcatDataset(cached_dataset_list)
    else:
        new_cached_dataset = cached_dataset_list[0] 

    vae.to(dtype=vae_dtype)

    return torch.utils.data.DataLoader(
                new_cached_dataset,
                batch_size=train_batch_size, 
                shuffle=shuffle,
                num_workers=2,
                persistent_workers=True,
                pin_memory=False,
                sampler=sampler
            )

def do_sanity_check(
    batch: Dict, 
    cache_latents: bool, 
    validation_pipeline: AnimationPipeline, 
    device: str, 
    image_finetune: bool=False,
    output_dir: str = "",
    dataset_id: int = 0
):
    pixel_values, texts = batch['pixel_values'].cpu(), batch["text_prompt"]
    
    if cache_latents:
        pixel_values = validation_pipeline.decode_latents(batch["pixel_values"].to(device))
        to_torch = torch.from_numpy(pixel_values)
        pixel_values = rearrange(to_torch, 'b c f h w -> b f c h w')
        
    if not image_finetune:
        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
            pixel_value = pixel_value[None, ...]
            text = f"{str(dataset_id)}_{text}"
            save_name = f"{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'-{idx}'}.mp4"
            save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{save_name}", rescale=not cache_latents)
    else:
        for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
            pixel_value = pixel_value / 2. + 0.5
            text = f"{str(dataset_id)}_{text}"
            save_name = f"{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'-{idx}'}.png"
            torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{save_name}")

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue
            
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params

def scale_loras(lora_list: list, scale: float, step=None, spatial_lora_num=None):
    
    # Assumed enumerator
    if step is not None and spatial_lora_num is not None:
        process_list = range(0, len(lora_list), spatial_lora_num)
    else:
        process_list = lora_list

    for lora_i in process_list:
        if step is not None:
            lora_list[lora_i].scale = scale
        else:
            lora_i.scale = scale

def get_spatial_latents(
        batch: Dict, 
        random_hflip_img: int, 
        cache_latents: bool,
        noisy_latents:torch.Tensor, 
        target: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler: DDPMScheduler
    ):
    ran_idx = torch.randint(0, batch["pixel_values"].shape[2], (1,)).item()
    use_hflip = random.uniform(0, 1) < random_hflip_img

    noisy_latents_input = None
    target_spatial = None

    if use_hflip:
        pixel_values_spatial = torchvision.transforms.functional.hflip(
            batch["pixel_values"][:, ran_idx, :, :, :] if not cache_latents else\
                batch["pixel_values"][:, :, ran_idx, :, :]
        ).unsqueeze(1)

        latents_spatial = (
            tensor_to_vae_latent(pixel_values_spatial, vae) if not cache_latents
            else
            pixel_values_spatial
        )

        noise_spatial = sample_noise(latents_spatial, 0,  use_offset_noise=use_offset_noise)
        noisy_latents_input = noise_scheduler.add_noise(latents_spatial, noise_spatial, timesteps)

        target_spatial = noise_spatial
    else:
        noisy_latents_input = noisy_latents[:, :, ran_idx, :, :]
        target_spatial = target[:, :, ran_idx, :, :]

    return noisy_latents_input, target_spatial, use_hflip

def create_ad_temporal_loss(
        model_pred: torch.Tensor, 
        loss_temporal: torch.Tensor, 
        target: torch.Tensor
    ):
    beta = 1
    alpha = (beta ** 2 + 1) ** 0.5

    ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()

    model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
    target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2)

    loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
    loss_temporal = loss_temporal + loss_ad_temporal

    return loss_temporal

def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,

    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    learning_rate_spatial: float = 1e-4,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,

    dataset_types: Tuple[str] = ('json'),
    motion_module_path: str = "",
    domain_adapter_path: str = "",

    random_hflip_img: float = -1,
    use_motion_lora_format: bool = True,
    single_spatial_lora: bool = False,
    lora_name: str = "motion_director_lora",
    lora_rank: int = 8,
    lora_unet_dropout: float = 0.1,
    train_temporal_lora: bool = True,
    target_spatial_modules: str = ["Transformer3DModel"],
    target_temporal_modules: str = ["TemporalTransformerBlock"],

    cache_latents: bool = False,
    cached_latent_dir=None,

    train_sample_validation: bool = True,
    device: str = 'cuda',
    use_text_augmenter: bool = False,
    use_lion_optim: bool = False,
    use_offset_noise: bool = False,
    *args,
    **kwargs
):
    check_min_version("0.10.0.dev0")
    
    if use_text_augmenter:
        print("Using random text augmentation")

    # Initialize distributed training
    num_processes   = 1        
    seed = global_seed
    torch.manual_seed(seed)
    
    # Logging folder
    if lora_name != "motion_director_lora":
        name = lora_name + f"_{name}"
        
    date_calendar = datetime.datetime.now().strftime("%Y-%m-%d")
    date_time = datetime.datetime.now().strftime("-%H-%M-%S")
    folder_name = "debug" if is_debug else name + date_time

    output_dir = os.path.join(output_dir, date_calendar, folder_name)

    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if not is_debug and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    lora_path = create_save_paths(output_dir)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler_kwargs.update({"steps_offset": 1})
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    del noise_scheduler_kwargs["steps_offset"]

    noise_scheduler_kwargs['beta_schedule'] = 'scaled_linear'
    train_noise_scheduler_spatial = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    
    # AnimateDiff uses a linear schedule for its temporal sampling
    noise_scheduler_kwargs['beta_schedule'] = 'linear'
    train_noise_scheduler = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    if kwargs.get("force_spatial_linear_scaling", True):
        train_noise_scheduler_spatial = train_noise_scheduler

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    # Freeze all models for LoRA training
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if not use_lion_optim:
        optimizer = torch.optim.AdamW
    else:
        optimizer = Lion
        learning_rate, learning_rate_spatial = map(lambda lr: lr / 10, (learning_rate, learning_rate_spatial))
        adam_weight_decay *= 10

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            if kwargs.get("force_temporal_xformers"):
                for module in unet.modules():
                    if module.__class__.__name__ == "VersatileAttention":
                        setattr(module, '_use_memory_efficient_attention_xformers', True)
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(device)
    text_encoder.to(device)

    # Get the training dataset
    train_dataset = get_train_dataset(dataset_types, train_data, tokenizer)

    if len(train_dataset) > 0:
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    else:
        train_dataset = train_dataset[0]
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    if cache_latents:
        torch.multiprocessing.set_start_method('spawn')
        train_dataloader = handle_cache_latents(
            cache_latents, 
            output_dir, 
            train_dataloader, 
            train_batch_size, 
            vae,
            cached_latent_dir=cached_latent_dir,
            sampler=None,
            device=device
        )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to(device)
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )

    validation_pipeline = load_weights(
        validation_pipeline, 
        motion_module_path=motion_module_path,
        adapter_lora_path=domain_adapter_path, 
        dreambooth_model_path=unet_checkpoint_path
    )

    validation_pipeline.enable_vae_slicing()
    validation_pipeline.to(device)

    unet.to(device=device)
    text_encoder.to(device=device)

    # Temporal LoRA
    if train_temporal_lora:
        # one temporal lora
        lora_manager_temporal = LoraHandler(use_unet_lora=True, unet_replace_modules=target_temporal_modules)
        
        unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
            True, unet, lora_manager_temporal.unet_replace_modules, 0,
            lora_path + '/temporal/', r=lora_rank)

        optimizer_temporal = optimizer(
            create_optimizer_params([param_optim(unet_lora_params_temporal, True, is_lora=True,
                                                 extra_params={**{"lr": learning_rate}}
                                                 )], learning_rate),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay
        )
    
        lr_scheduler_temporal = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_temporal,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
    else:
        lora_manager_temporal = None
        unet_lora_params_temporal, unet_negation_temporal = [], []
        optimizer_temporal = None
        lr_scheduler_temporal = None

    # Spatial LoRAs
    if single_spatial_lora:
        spatial_lora_num = 1
    else:
        # one spatial lora for each video
        spatial_lora_num = train_dataset.__len__()

    lora_managers_spatial = []
    unet_lora_params_spatial_list = []
    optimizer_spatial_list = []
    lr_scheduler_spatial_list = []

    for i in range(spatial_lora_num):
        lora_manager_spatial = LoraHandler(use_unet_lora=True, unet_replace_modules=target_spatial_modules)
        lora_managers_spatial.append(lora_manager_spatial)
        unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
            True, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
            lora_path + '/spatial/', r=lora_rank)

        unet_lora_params_spatial_list.append(unet_lora_params_spatial)

        optimizer_spatial = optimizer(
            create_optimizer_params([param_optim(unet_lora_params_spatial, True, is_lora=True,
                                                 extra_params={**{"lr": learning_rate_spatial}}
                                                 )], learning_rate_spatial),
            lr=learning_rate_spatial,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay
        )

        optimizer_spatial_list.append(optimizer_spatial)

        # Scheduler
        lr_scheduler_spatial = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_spatial,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        lr_scheduler_spatial_list.append(lr_scheduler_spatial)

        unet_negation_all = unet_negation_spatial + unet_negation_temporal

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    
    ### <<<< Training <<<< ###
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            spatial_scheduler_lr = 0.0
            temporal_scheduler_lr = 0.0

            # Handle Lora Optimizers & Conditions
            for optimizer_spatial in optimizer_spatial_list:
                optimizer_spatial.zero_grad(set_to_none=True)

            if optimizer_temporal is not None:
                optimizer_temporal.zero_grad(set_to_none=True)

            if train_temporal_lora:
                mask_temporal_lora = False
            else:
                mask_temporal_lora = True

            mask_spatial_lora =  random.uniform(0, 1) < 0.2 and not mask_temporal_lora

            if cfg_random_null_text:
                batch["text_prompt"] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch["text_prompt"]]
            
            if use_text_augmenter:
                random.seed()
                txt_idx = random.randint(0, len(augment_text_list) - 1)
                augment_text = augment_text_list[txt_idx]
                
                batch['text_prompt'] = [
                    f"{augment_text} {prompt}" for prompt in batch['text_prompt']
                ]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                for _idx, _batch in enumerate(tqdm(train_dataloader, desc="Dataset sanity check...")):
                    do_sanity_check(
                        _batch, 
                        cache_latents, 
                        validation_pipeline, 
                        device, 
                        output_dir=output_dir, 
                        dataset_id=_idx
                    )
                    if _idx > 10:
                        break

            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(device)
            video_length = pixel_values.shape[2]
            bsz = pixel_values.shape[0]       

            # Sample a random timestep for each video
            timesteps = torch.randint(0, train_noise_scheduler.config.num_train_timesteps, (bsz,), device=pixel_values.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            latents = tensor_to_vae_latent(pixel_values, vae) if not cache_latents else pixel_values
            noise = sample_noise(latents, 0, use_offset_noise=use_offset_noise)
            target = noise         

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text_prompt'], 
                    max_length=tokenizer.model_max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(pixel_values.device)
                
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                if mask_spatial_lora:
                    loras = extract_lora_child_module(unet, target_replace_module=target_spatial_modules)
                    scale_loras(loras, 0.)
                    loss_spatial = None
                else:
                    loras = extract_lora_child_module(unet, target_replace_module=target_spatial_modules)
                    if spatial_lora_num == 1:
                        scale_loras(loras, 1.0)
                    else:
                        scale_loras(loras, 0.)
                        scale_loras(loras, 1.0, step=step, spatial_lora_num=spatial_lora_num)

                    loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                    if len(loras) > 0:
                        scale_loras(loras, 0.)
                    
                    ### >>>> Spatial LoRA Prediction >>>> ###
                    noisy_latents = train_noise_scheduler_spatial.add_noise(latents, noise, timesteps)
                    noisy_latents_input, target_spatial, use_hflip = get_spatial_latents(
                        batch, 
                        random_hflip_img, 
                        cache_latents,
                        noisy_latents,
                        target,
                        timesteps,
                        train_noise_scheduler_spatial
                    )

                    if use_hflip:
                        model_pred_spatial = unet(noisy_latents_input, timesteps,
                                                encoder_hidden_states=encoder_hidden_states).sample
                        loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                                target_spatial[:, :, 0, :, :].float(), reduction="mean")
                    else:
                        model_pred_spatial = unet(noisy_latents_input.unsqueeze(2), timesteps,
                                                encoder_hidden_states=encoder_hidden_states).sample
                        loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                                target_spatial.float(), reduction="mean")

                if mask_temporal_lora:
                    loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                    scale_loras(loras, 0.)
                    
                    loss_temporal = None
                else:
                    loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                    scale_loras(loras, 1.0)
                    
                    ### >>>> Temporal LoRA Prediction >>>> ###
                    noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                    
                    loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss_temporal = create_ad_temporal_loss(model_pred, loss_temporal, target)

                # Backpropagate
                if not mask_spatial_lora:
                    scaler.scale(loss_spatial).backward(retain_graph=True)
                    if spatial_lora_num == 1:
                        scaler.step(optimizer_spatial_list[0])
                        
                    else:
                        # https://github.com/nerfstudio-project/nerfstudio/pull/1919
                        if any(
                            any(p.grad is not None for p in g["params"]) for g in optimizer_spatial_list[step].param_groups
                        ):
                            scaler.step(optimizer_spatial_list[step])
                        
                if not mask_temporal_lora and train_temporal_lora:
                    scaler.scale(loss_temporal).backward()
                    scaler.step(optimizer_temporal)
                           
                if spatial_lora_num == 1:
                    lr_scheduler_spatial_list[0].step()
                    spatial_scheduler_lr = lr_scheduler_spatial_list[0].get_lr()[0]
                else:
                    lr_scheduler_spatial_list[step].step()
                    spatial_scheduler_lr = lr_scheduler_spatial_list[step].get_lr()[0]
                    
                if lr_scheduler_temporal is not None:
                    lr_scheduler_temporal.step()
                    temporal_scheduler_lr = lr_scheduler_temporal.get_lr()[0]
            
            scaler.update()  
            progress_bar.update(1)
            global_step += 1
            
            # Wandb logging
            if not is_debug and use_wandb:
                loss = (
                    loss_temporal if loss_spatial is None else \
                        loss_temporal + loss_spatial
                )
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if global_step % checkpointing_steps == 0:
                import copy
                
                # We do this to prevent VRAM spiking / increase from the new copy
                validation_pipeline.to('cpu')

                lora_manager_spatial.save_lora_weights(
                    model=copy.deepcopy(validation_pipeline), 
                    save_path=lora_path+'/spatial', 
                    step=global_step,
                    use_safetensors=True,
                    lora_rank=lora_rank,
                    lora_name=lora_name + "_spatial"
                )

                if lora_manager_temporal is not None:
                    lora_manager_temporal.save_lora_weights(
                        model=copy.deepcopy(validation_pipeline), 
                        save_path=lora_path+'/temporal', 
                        step=global_step,
                        use_safetensors=True,
                        lora_rank=lora_rank,
                        lora_name=lora_name + "_temporal",
                        use_motion_lora_format=use_motion_lora_format
                    )

                validation_pipeline.to(device)

            # Periodically validation
            if (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                validation_seed = getattr(validation_data, 'seed', -1)

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed if validation_seed == -1 else validation_seed)
                
                if not train_sample_validation:
                    if not isinstance(train_data.sample_size, int):
                        height, width = train_data.sample_size[:2]
                    else:
                        height, width = [train_data.sample_size] * 2
                else:
                    if all(['resized_h'in batch, 'resized_w' in batch]):
                        height, width = batch["resized_h"], batch['resized_w'] 
                    else:
                        height, width = [512] * 2

                prompts = (
                    validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) \
                        else validation_data.prompts
                )
                
                with torch.cuda.amp.autocast(enabled=True):
                    if gradient_checkpointing:
                        unet.disable_gradient_checkpointing()

                    loras = extract_lora_child_module(
                        unet, 
                        target_replace_module=target_spatial_modules
                    )
                    scale_loras(loras, validation_data.spatial_scale)
                    
                    with torch.no_grad():
                        unet.eval()
                        for idx, prompt in enumerate(prompts):
                            if len(prompt) == 0:
                                prompt = batch['text_prompt']
                            print(prompt)
                            if not image_finetune:
                                sample = validation_pipeline(
                                    prompt,
                                    generator    = generator,
                                    video_length = train_data.sample_n_frames,
                                    height       = height,
                                    width        = width,
                                    **validation_data,
                                ).videos
                                save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                                samples.append(sample)
                                
                            else:
                                sample = validation_pipeline(
                                    prompt,
                                    generator           = generator,
                                    height              = height,
                                    width               = width,
                                    num_inference_steps = validation_data.get("num_inference_steps", 25),
                                    guidance_scale      = validation_data.get("guidance_scale", 8.),
                                ).images[0]
                                sample = torchvision.transforms.functional.to_tensor(sample)
                                samples.append(sample)
                        unet.train()

                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logging.info(f"Saved samples to {save_path}")
            
            logs = {
                "Temporal Loss": loss_temporal.detach().item(),
                "Temporal LR": temporal_scheduler_lr, 
                "Spatial Loss": loss_spatial.detach().item() if loss_spatial is not None else 0,
                "Spatial LR": spatial_scheduler_lr
            }
            progress_bar.set_postfix(**logs)
            
            if gradient_checkpointing:
                unet.enable_gradient_checkpointing()

            if global_step >= max_train_steps:
                break
                     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)
      
    if getattr(config, "simple_mode", False):
        config = get_simple_config(config)
    
    main(name=name, use_wandb=args.wandb, **config)

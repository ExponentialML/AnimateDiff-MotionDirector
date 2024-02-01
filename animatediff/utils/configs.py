import sys

from omegaconf import OmegaConf
from types import SimpleNamespace

QUALITY_TYPES = ["low", "preferred", "best"]

def create_quality_config(
    width: int, 
    height: int, 
    sample_width: int = 384, 
    sample_height: int = 384, 
    use_bucketing: bool = False,
    lora_rank: int = 2
):
    config = dict(
        width=width,
        height=height,
        use_bucketing=use_bucketing,
        sample_size=(
            sample_height if sample_height!= 0 else 256, 
            sample_width if sample_width != 0 else 256
        ),
        lora_rank=lora_rank
    )

    return SimpleNamespace(**config)

def set_train_data(config: SimpleNamespace, quality_config: SimpleNamespace):
        train_data_map = ["sample_size", "width", "height", "sample_size", "use_bucketing"]
        
        # Set LoRA Rank fallback
        setattr(config, 'lora_rank', getattr(quality_config, 'lora_rank', 8))
        
        for train_setting in train_data_map:
            if getattr(config.train_data, 'manual_sample_size', False) and \
                train_setting == "sample_size":
                continue

            setattr(config.train_data, train_setting, getattr(quality_config, train_setting))

def set_single_video_args(config: SimpleNamespace, simple_config: SimpleNamespace):
    config.dataset_types = ["single_video"]

    single_data_map = [
        ("max_chunks", "max_chunks"),
        ("single_video_path", "path"),
        ("sample_start_idx", "start_time"),
        ("single_video_prompt", "training_prompt"),
    ]
    
    for single_data_key, simple_config_key in single_data_map:
        if simple_config_key == "max_chunks":
            setattr(
                config.train_data, 
                single_data_key, 
                getattr(simple_config.video, simple_config_key, sys.maxsize)
            )
            continue

        setattr(config.train_data, single_data_key, getattr(simple_config.video, simple_config_key))

def set_folder_of_videos_args(config: SimpleNamespace, simple_config: SimpleNamespace):
    config.dataset_types = ["folder"]

    folder_data_map = [
        ("max_chunks", "max_chunks"),
        ("path", "path"),
        ("single_video_prompt", "training_prompt"),
        ("fallback_prompt", "training_prompt"),
        ("prompts", "validation_prompt")
    ]

    for folder_data_key, simple_config_key in folder_data_map:
        if simple_config_key == "max_chunks":
            setattr(
                config.train_data, 
                folder_data_key, 
                getattr(simple_config.video, simple_config_key, sys.maxsize)
            )
            continue

        setattr(config.train_data, folder_data_key, getattr(simple_config.video, simple_config_key))

def build_quality_configs():
    LowQualityConfig = create_quality_config(256, 256, 512, 512, lora_rank=32)
    PreferredConfig = create_quality_config(384, 384, 384, 384, use_bucketing=True, lora_rank=64)
    BestQualityConfig = create_quality_config(512, 512, 512, 512, use_bucketing=True, lora_rank=64)

    quality_configs = {"low": LowQualityConfig, "preferred": PreferredConfig, "best": BestQualityConfig}

    return quality_configs

def get_simple_config(config: OmegaConf):
    simple_config = None
    quality_configs = build_quality_configs()
    
    try:
        checkpoints_map = [
            "pretrained_model_path",
            "motion_module_path",
            "unet_checkpoint_path",
            "domain_adapter_path"
        ]

        simple_config = config
        config = OmegaConf.load(config.training_config)
        config.lora_name = simple_config.save_name
        
        for checkpoint_key in checkpoints_map:
            setattr(config, checkpoint_key, getattr(simple_config, checkpoint_key))

    except Exception as e:
        raise ValueError("Could not load training config", e)

    if simple_config.quality.lower() not in QUALITY_TYPES:
        raise ValueError(f"Quality must be the following: {QUALITY_TYPES}")

    quality_config = quality_configs.get(simple_config.quality.lower())    
    set_train_data(config, quality_config)
    
    if simple_config.mode_type == "single_video":
        set_single_video_args(config, simple_config)
    elif simple_config.mode_type == "folder":
        set_folder_of_videos_args(config, simple_config)
    else: 
        raise ValueError(f"{simple_config.mode_type} not imlemented. Choose 'single_video' or 'folder'")

    config.validation_data.prompts[0] = simple_config.video.validation_prompt

    return config


import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
from .bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat

TRAIN_DATA_VARS = ['train_data', 'frames', 'image_dir', 'video_files']
VID_TYPES = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids
    return prompt_ids

def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

    
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices

def process_video(
    vid_path, 
    use_bucketing, 
    w, 
    h, 
    get_frame_buckets, 
    get_frame_batch, 
    callback=None
):
    resized_h = None
    resized_w = None

    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize, height, width = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)
        resized_h, resized_w = height, width       
    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)
        resized_h, resized_w = w, h

    if callback is not None:
        callback(resized_h, resized_w)

    return video, vr


class DatasetProcessor(object):
    def __init__(self, cond_processor=None, cond_processor_kwargs={}):
        self.condition_processor_model_loaded = False
        self.condition_processor_kwargs = cond_processor_kwargs
        self.condition_processor_name = ""
        self.condition_enabled = False
        self.resized_w = 0
        self.resized_h = 0

    def get_frame_range(self, vr):
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            self.frame_step, 
            self.n_sample_frames
        )

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape        
        width, height = sensible_buckets(
            self.width, 
            self.height, 
            w, 
            h, 
            extra_simple=False,
            min_size=256
        )
        resize = T.transforms.Resize(
            (height, width), 
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return resize, height, width

    def set_resize_props(self, resized_h, resized_w, *args, **kwargs):
        self.resized_h = resized_h
        self.resized_w = resized_w

    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch,
                callback=self.set_resize_props
            )
        
        return video, vr 

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def create_video_chunks(
        self, 
        video_path: str, 
        fps: int,
        frame_step: int, 
        n_sample_frames: int,
        max_chunks: int,
        start_idx: int
    ):
        # Create a list of frames separated by sample frames
        # [(1,2,3), (4,5,6), ...]
        vr = decord.VideoReader(video_path)
        
        frame_step =  min(self.get_avg_fps(vr, fps), 3) if fps > 0 else frame_step
        vr_range = range(start_idx, len(vr), frame_step)

        frames = list(self.chunk(vr_range, n_sample_frames))

        # Delete any list that contains an out of range index.
        frames = list(
            filter(lambda x: len(x) == n_sample_frames, frames)
        )

        return frames[:self.max_video_clips(frames, max_chunks)]

    def get_avg_fps(self, vr: decord.VideoReader, fps: int = 0):
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / fps))
        every_nth_frame = min(len(vr), every_nth_frame)

        return every_nth_frame

    def max_video_clips(self, frames: int, max_chunks: int):
        return len(frames) if max_chunks == 0 else max_chunks

    # Inspired by the VideoMAE repository.
    def normalize_input(
        self,
        item, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
        use_simple_norm=False
    ):
        if item.dtype == torch.uint8 and not use_simple_norm:
            import warnings
            warnings.warn("Using norm based off of ImageNet.")

            item = rearrange(item, 'f c h w -> f h w c')
            
            item = item.float() / 255.0
            mean = torch.tensor(mean)
            std = torch.tensor(std)

            out = rearrange((item - mean) / std, 'f h w c -> f c h w')
            
            return out
        else:
            item = item.float() / 255.
            item = torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)(item)

            return item

    def _example(self, item, prompt_ids, prompt):
        example = {
            "pixel_values": self.normalize_input(item, use_simple_norm=True),
            "resized_h": self.resized_h,
            "resized_w": self.resized_w,
            "prompt_ids": prompt_ids,
            "text_prompt": prompt,
            'dataset': self.__getname__(),
        }

        self.resized_h, self.resized_w = 0, 0
        
        return example

# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoJsonDataset(DatasetProcessor, Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 1,
            frame_step: int = 1,
            json_path: str ="",
            json_data = None,
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            use_bucketing: bool = False,
            condition_processor = None,
            **kwargs
    ):
        DatasetProcessor.__init__(self, condition_processor, kwargs.get('cond_processor_kwargs', {}))
        self.vid_types = VID_TYPES
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key
        self.train_data = self.load_from_json(json_path, json_data)

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.frame_step = frame_step

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        clip_path = nested_data['clip_path'] if 'clip_path' in nested_data else None
        
        extended_data.append({
            self.vid_data_key: data[self.vid_data_key],
            'frame_index': nested_data['frame_index'],
            'prompt': nested_data['prompt'],
            'clip_path': clip_path
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                print(f"Loading JSON from {path}")
                json_data = json.load(jpath)

                return self.build_json(json_data)

        except:
            self.train_data = []
            print("Non-existant JSON path. Skipping.")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def train_data_batch(self, index):

        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index] and \
            self.train_data[index]['clip_path'] is not None:

            vid_data = self.train_data[index]

            clip_path = vid_data['clip_path']
            
            # Get video prompt
            prompt = vid_data['prompt']

            video, _ = self.process_video_wrapper(clip_path)

            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids

         # Assign train data
        train_data = self.train_data[index]
        
        # Get the frame of the current index.
        self.sample_start_idx = train_data['frame_index']
        
        # Initialize resize
        resize = None

        video, vr = self.process_video_wrapper(train_data[self.vid_data_key])

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.train_data is not None:
            video, prompt, prompt_ids = self.train_data_batch(index)

        return self._example(video, prompt_ids, prompt)


class SingleVideoDataset(DatasetProcessor, Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            fps: int = 0,
            frame_step: int = 1,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            condition_processor = None,
            max_chunks: int = 0,
            sample_start_idx: int = 0,
            **kwargs
    ):
        DatasetProcessor.__init__(self, condition_processor, kwargs.get('cond_processor_kwargs', {}))
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.fps = fps
        self.frame_step = frame_step
        self.max_chunks = max_chunks
        self.sample_start_idx = sample_start_idx

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt
        self.frames = self.create_video_chunks(
            single_video_path, 
            fps, 
            frame_step, 
            n_sample_frames, 
            max_chunks,
            sample_start_idx
        )

        self.width = width
        self.height = height

    def get_frame_batch(self, vr, resize=None):
        index = self.index

        frames = vr.get_batch(self.frames[self.index])
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: 
            video = resize(video)
        return video

    def get_prompt(self):
        vid_ext = self.single_video_path.split(".")[-1]
        video_path = self.single_video_path

        maybe_text_file = video_path.replace(f".{vid_ext}", ".txt")
        
        if os.path.exists(maybe_text_file):
            with open(maybe_text_file, "r") as f:
                prompt = f.read()
        else:
            prompt = self.single_video_prompt

        return prompt

    def single_video_batch(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, _ = self.process_video_wrapper(train_data)

            prompt = self.get_prompt()
            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): 
        return 'single_video'

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        video, prompt, prompt_ids = self.single_video_batch(index)
        return self._example(video, prompt_ids, prompt)
    
class ImageDataset(DatasetProcessor, Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        condition_processor = None,
        **kwargs
    ):
        DatasetProcessor.__init__(self, condition_processor, kwargs.get('cond_processor_kwargs', {}))
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing

        self.image_dir = self.get_images_list(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h, extra_simple=False)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=1)

        prompt = get_text_prompt(
            file_path=train_data,
            text_prompt=self.single_img_prompt,
            fallback_prompt=self.fallback_prompt,
            ext_types=self.img_types,  
            use_caption=True
        )
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        img, prompt, prompt_ids = self.image_batch(index)
    
        return self._example(img, prompt_ids, prompt)

# NOTE: This is currently unused in this repository. All videos are processed with SingleVideoDataset.
# If you are doing folder based training, all single videos are concatenated into a single dataset using ConcatDataset.
# The VideoFolderDataset class is still usable, but must be manually set and modified in your training script.
class VideoFolderDataset(DatasetProcessor, Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        condition_processor = None,
        sample_start_idx: int = 0,
        **kwargs
    ):
        DatasetProcessor.__init__(self, condition_processor, kwargs.get('cond_processor_kwargs', {}))
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.sample_start_idx = sample_start_idx
        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video, vr
        
    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        
        video, _ = self.process_video_wrapper(self.video_files[index])

        if os.path.exists(self.video_files[index].replace(".mp4", ".txt")):
            with open(self.video_files[index].replace(".mp4", ".txt"), "r") as f:
                prompt = f.read()
        else:
            prompt = self.fallback_prompt
        
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return self._example(video[0], prompt_ids, prompt)

class CachedDataset(DatasetProcessor, Dataset):
    def __init__(self, cache_dir: str = ''):
        DatasetProcessor.__init__(self)
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cpu')

        return cached_latent

class ConcatInterleavedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.train_data_vars = TRAIN_DATA_VARS
        
        self.interleave_datasets()

    def get_parent_dataset(self):

        # There's a chance that the subset images may be bigger than the video if doing text training.
        # If it has the attribute "is_subset", we can simply ignore it to ensure it isn't the biggest 
        # length.
        dataset_lengths = [d.__len__() if not hasattr(d, 'is_subset') else 0 for d in self.datasets]
        max_dataset_index = dataset_lengths.index(max(dataset_lengths))

        parent_dataset = self.datasets[max_dataset_index]

        return parent_dataset, max_dataset_index

    def process_dataset(self, dataset):
        processed_dataset = []
        train_data_var_name = self.get_dataset_data_var_name(dataset)[0]
        train_data_var = getattr(dataset, train_data_var_name)

        for idx, item in enumerate(train_data_var):
            if isinstance(item, dict) and 'idx_modulo' in item:
                ref_idx = item['idx_modulo']
                already_processed_item = processed_dataset[ref_idx]

                # Dataset items are assumed to be of type Dict
                already_processed_item['reference_idx'] = ref_idx
                processed_dataset.append(already_processed_item)
            else:
                processed_dataset.append(dataset[idx])
        
        return processed_dataset

    def get_dataset_data_var_name(self, dataset):
        return [v for v in self.train_data_vars if v in dataset.__dict__.keys()]

    def create_data_val_dict(self, val, idx, length, idx_modulo):
        return dict(
            value=val,
            idx=idx,
            length=length,
            idx_modulo=idx_modulo
        )

    def interleave_datasets(self):
        parent_dataset, parent_dataset_index = self.get_parent_dataset()
        child_datasets = self.datasets.copy()
        child_datasets.pop(parent_dataset_index)

        parent_dataset_length = parent_dataset.__len__()

        for dataset in child_datasets:
            if dataset.__len__() <= 0:
                del dataset
                continue
            
            var_name = self.get_dataset_data_var_name(dataset)
            var_name = var_name[0] if len(var_name) == 1 else None

            if var_name is None:
                continue
            
            original_dataset_length = dataset.__len__()
            
            train_data_var = getattr(dataset, var_name)
            train_data_var *= parent_dataset_length
            new_train_data_val = train_data_var[:parent_dataset_length]

            # Do this to reference items that were already accessed.
            # Since some __getitem__ functions are heavy (numpy computations, video reads, etc.),
            # we want to avoid performing the same expensive function multiple times.
            # We simply point to the corresponding index so that when we interleave, we can just copy the __getitem__ result.
            for i, val in enumerate(new_train_data_val):
                if i >= original_dataset_length:
                    clamped_idx = i % original_dataset_length
                    new_train_data_val[i] = self.create_data_val_dict(
                        val, 
                        i, 
                        original_dataset_length, 
                        clamped_idx
                    )
                    
            setattr(dataset, var_name, new_train_data_val)

        from itertools import chain

        print("Interleaving Datasets. Please wait...")
        train_datasets = [parent_dataset] + child_datasets

        # Zip all of the items in the datasets. We do this to __get_item__ all of our data. 
        # Example (d == Dataset): [(d1_item1, d2_item1, d3_item1), (d1_item2, d2_item2, d3_item2), (...)]
        interleave_datasets = zip(*[self.process_dataset(d) for d in train_datasets])

        # Now we flatten it as a new Dataset iterable Dataset to be concatenated.
        # Example: [d1_item1, d2_item1, d3_item1, d2_item1, d2_item2, d2_item3, ...]
        InterLeavedDataset = list(chain(*interleave_datasets))
        self.datasets = InterLeavedDataset

        print("Finished interleaving datasets.")

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]

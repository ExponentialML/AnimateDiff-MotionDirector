import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
        dataset_norm=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.color_jitter = color_jitter
        self.h_flip = h_flip
        self.tokenizer = tokenizer
        self.resize = resize
        self.dataset_norm = dataset_norm

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        
        self.image_transforms = self.compose()
        self.normalized_mean_std = self.get_dataset_norm(class_data_root)


    def gather_norm(self, img, mean=None, std=None):
        channels = img.shape[0]

        if all(x is None for x in [mean, std]):
            mean, std = torch.zeros(channels), torch.zeros(channels)

        for i in range(channels):
            mean[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()    
        
        return mean, std

    def get_dataset_norm(self, class_data_root):
        if self.dataset_norm:
            imgs_to_process = self.instance_images_path
            
            if class_data_root is not None:
                imgs_to_process += self.class_images_path

            mean = None
            std = None
            
            for img in tqdm(imgs_to_process, desc="Processing image normalization..."):
                img = Image.open(img).convert("RGB")
                img = self.image_transforms(img)
                
                mean, std = self.gather_norm(img, mean, std)

            mean.div_(len(imgs_to_process))
            std.div_(len(imgs_to_process))

            print(f"Dataset mean and std are: {mean}, {std}")
            
            return mean, std
        else:
            return [0.5], [0.5]

    def compose(self):
        img_transforms = []

        if self.resize:
            img_transforms.append(
                transforms.Resize(
                    (self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if self.center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if self.color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if self.h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        return transforms.Compose([*img_transforms, transforms.ToTensor()])
        
    def image_transform(self, img): 
        img_composed = self.image_transforms(img)

        if not self.dataset_norm:
            mean, std = self.gather_norm(img_composed)
            mean, std = [0.5], [0.5]
        else:
            mean, std = self.normalized_mean_std

        return transforms.Normalize(mean, std)(img_composed)

    def open_img(self, index, folder):
        img_path =  folder[index % self.num_instance_images]
        img = Image.open(img_path)

        if not img.mode == "RGB":
            img = img.convert("RGB")

        return img, str(img_path).split("/")[-1]

    def tokenize_prompt(self, prompt):
        return self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids


    def get_train_sample(self, index, example, base_name, folder, prompt):
        image, img_name = self.open_img(index, self.instance_images_path)
        example[f"{base_name}_images"] = self.image_transform(image)
        example[f"{base_name}_prompt_ids"] = self.tokenize_prompt(prompt)
        example[f"{base_name}_prompt"] = prompt
        example[f"{base_name}_img_name"] = img_name

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        self.get_train_sample(
            index, 
            example, 
            "instance", 
            self.instance_images_path, 
            self.instance_prompt
        )

        if self.class_data_root:
            self.get_train_sample(
                index, 
                example, 
                "class", 
                self.class_images_path, 
                self.class_prompt
            )
        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
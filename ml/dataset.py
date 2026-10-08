import os
import io
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .vocab import TextTokenizer

LABEL_MAP = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2
}

IDX_TO_LABEL = {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}

LABEL_TO_IDX = LABEL_MAP

# Image transformation pipeline
DEFAULT_IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_source, transform=DEFAULT_IMAGE_TRANSFORM) -> torch.Tensor:
    """Preprocesses a PIL Image, file path, or raw bytes into normalized [3, 224, 224] Tensor."""
    if isinstance(image_source, str):
        image = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    return transform(image)


class MultimodalDataset(Dataset):
    """
    Dataset class for paired text-image sentiment samples.
    """
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: TextTokenizer,
        image_root_dir: str = "",
        max_seq_len: int = 64,
        transform=DEFAULT_IMAGE_TRANSFORM
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_root_dir = image_root_dir
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]

        # 1. Text tokenization
        text = item.get("text", "")
        token_ids = self.tokenizer.encode(text, max_len=self.max_seq_len)
        text_tensor = torch.tensor(token_ids, dtype=torch.long)

        # 2. Image loading and transform
        image_path = item.get("image", "")
        if self.image_root_dir and not os.path.isabs(image_path):
            full_path = os.path.join(self.image_root_dir, image_path)
        else:
            full_path = image_path

        if os.path.exists(full_path):
            img_tensor = preprocess_image(full_path, transform=self.transform)
        else:
            # Fallback blank tensor if image path not found
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 3. Label encoding
        label_str = item.get("label", "neutral")
        label_idx = LABEL_MAP.get(label_str, 1)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return text_tensor, img_tensor, label_tensor

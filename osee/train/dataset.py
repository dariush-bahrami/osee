import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Iterable
from transformers import TrOCRProcessor


class TrOCRDataset(Dataset):
    def __init__(
        self,
        data: Iterable[tuple[Image.Image, str]],
        processor: TrOCRProcessor,
        max_target_length: int,
    ):
        self.data = data
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        image, text = self.data[index]
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding

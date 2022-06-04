from enum import Enum
from pathlib import Path

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class ModelSize(Enum):
    SMALL = 0
    BASE = 1
    LARGE = 2


class ModelType(Enum):
    PRINTED = 0
    HANDWRITTEN = 1


def get_model_name(model_size: ModelSize, model_type: ModelType):
    parts = ["microsoft/trocr"]
    parts.append(model_size.name.lower())
    parts.append(model_type.name.lower())
    return "-".join(parts)


class OCRModel:
    def __init__(self, model: VisionEncoderDecoderModel, processor: TrOCRProcessor):
        self.model = model
        self.processor = processor

    def __call__(self, image: Image) -> str:
        image = image.convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_text[0]

    def save(self, save_directory: Path) -> "OCRModel":
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        return self

    def __apply_configs(self) -> "OCRModel":
        # set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 10
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        return self

    @classmethod
    def load_local_model(cls, save_directory: Path) -> "OCRModel":
        model = VisionEncoderDecoderModel.from_pretrained(save_directory)
        processor = TrOCRProcessor.from_pretrained(save_directory)
        return cls(model, processor).__apply_configs()

    @classmethod
    def load_from_hub(cls, model_size: ModelSize, model_type: ModelType) -> "OCRModel":
        model_name = get_model_name(model_size, model_type)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name)
        return cls(model, processor).__apply_configs()

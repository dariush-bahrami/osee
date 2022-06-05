from pathlib import Path
from typing import Iterable, Tuple

from datasets import load_metric
from PIL import Image
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..model import OCRModel
from .dataset import TrOCRDataset


def get_seq2seq_trainer(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    train_dataset: TrOCRDataset,
    eval_dataset: TrOCRDataset,
    batch_size: int,
    output_dir: Path,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    max_steps: int,
    dataloader_num_workers: int,
) -> Seq2SeqTrainer:
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        output_dir=output_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        max_steps=max_steps,
        dataloader_num_workers=dataloader_num_workers,
    )

    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    return trainer


class OCRTrainer:
    def __init__(
        self,
        model: OCRModel,
        train_data: Iterable[Tuple[Image.Image, str]],
        eval_data: Iterable[Tuple[Image.Image, str]],
        max_target_length: int,
        batch_size: int,
        output_dir: Path,
        logging_steps: int,
        save_steps: int,
        eval_steps: int,
        max_steps: int,
        dataloader_num_workers: int,
    ):
        self.model = model
        self.train_dataset = TrOCRDataset(
            train_data,
            self.model.processor,
            max_target_length,
        )
        self.eval_dataset = TrOCRDataset(
            eval_data,
            self.model.processor,
            max_target_length,
        )
        self.batch_size = batch_size

        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.max_steps = max_steps
        self.dataloader_num_workers = dataloader_num_workers
        self.hugging_face_trainer = get_seq2seq_trainer(
            self.model.model,
            self.model.processor,
            self.train_dataset,
            self.eval_dataset,
            self.batch_size,
            str(self.output_dir),
            self.logging_steps,
            self.save_steps,
            self.eval_steps,
            self.max_steps,
            self.dataloader_num_workers,
        )

    def train(self):
        self.hugging_face_trainer.train()

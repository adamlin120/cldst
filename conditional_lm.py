import logging
import os
import json
from typing import Dict, List
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import (
    Trainer,
    seed_everything,
    loggers,
    LightningModule,
    LightningDataModule,
    TrainResult,
    EvalResult,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from preprocess_multiwoz_data import SLOT_SEP, SLOT_NAME_SEP, SLOT_VALUE_SEP


logging.basicConfig(level=logging.INFO)


IGNORE_INDEX = -100
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
BELIEF = "<BELIEF>"
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": BOS,
    "eos_token": EOS,
    "pad_token": PAD,
    "additional_special_tokens": [BELIEF, SLOT_SEP, SLOT_NAME_SEP, SLOT_VALUE_SEP],
}


class ConditionalLM(LightningModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        tokenizer_class = (
            GPT2Tokenizer if self.hparams.gpt2_tokenizer else BertTokenizer
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            self.hparams.model_checkpoint, do_lower_case=True
        )
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_checkpoint)
        # Add special tokens if they are not already added
        add_special_tokens_(self.model, self.tokenizer)

    def forward(self, *args, **kwargs) -> CausalLMOutputWithPast:
        return self.model.forward(return_dict=True, *args, **kwargs)

    def training_step(self, batch, batch_idx) -> TrainResult:
        output = self.forward(**batch)
        result = TrainResult(
            minimize=output.loss,
        )
        result.log("train_loss", output.loss.detach(), prog_bar=True, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx) -> EvalResult:
        output = self.forward(**batch)
        result = EvalResult(checkpoint_on=output.loss, early_stop_on=output.loss)
        result.log(
            "val_loss",
            output.loss,
            prog_bar=True,
        )
        return result

    def test_step(self, batch, batch_idx) -> EvalResult:
        output = self.forward(**batch)
        result = EvalResult(checkpoint_on=output.loss, early_stop_on=output.loss)
        result.log(
            "test_loss",
            output.loss,
            prog_bar=True,
        )
        return result

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.hparams.lr, correct_bias=True
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="models/CDial-GPT2_LCCC-base/",
            help="Dir path to pretrained model",
        )
        parser.add_argument(
            "--gpt2_tokenizer",
            action="store_true",
            help="use gpt2 tokenizer instead of bert tokenizer",
        )
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        return parser


def add_special_tokens_(model, tokenizer):
    """Add special tokens to the tokenizer and the model if they have not
    already been added."""
    orig_num_tokens = len(tokenizer)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


class MultiwozDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: BertTokenizer,
        max_len: int = 512,
        testing: bool = False,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.testing = testing

        self.data = json.loads(self.path.read_text())
        self.turn_ids: List[str] = list(self.data.keys())
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(PAD)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: str) -> Dict[str, List[int]]:
        turn_id = self.turn_ids[index]
        turn = self.data[turn_id]
        instance = build_input_from_segments(
            turn["history"],
            turn.get("belief", "") if not self.testing else "",
            self.tokenizer,
        )
        return instance

    def collate_fn(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        out = {
            "input_ids": pad_truncate_sequence(
                [i["input_ids"] for i in batch], self.pad_token_id, self.max_len
            ),
            "labels": pad_truncate_sequence(
                [i["input_ids"] for i in batch], IGNORE_INDEX, self.max_len
            ),
            "attention_mask": pad_truncate_attention_mask(
                [i["attention_mask"] for i in batch], self.max_len
            ),
        }
        print(out)
        return out


class MultiWOZDataModule(LightningDataModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        tokenizer_class = (
            GPT2Tokenizer if self.hparams.gpt2_tokenizer else BertTokenizer
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            self.hparams.model_checkpoint, do_lower_case=True
        )
        self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.datasets: Dict[str, MultiwozDataset] = {}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            for split in ["train", "val"]:
                self.datasets[split] = MultiwozDataset(
                    Path(self.hparams.data_dir) / f"{split}.json",
                    self.tokenizer,
                    self.hparams.max_len,
                )

        if stage == "test" or stage is None:
            self.datasets["test"] = MultiwozDataset(
                Path(self.hparams.data_dir) / "test.json",
                self.tokenizer,
                self.hparams.max_len,
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            collate_fn=self.datasets["train"].collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            collate_fn=self.datasets["val"].collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            collate_fn=self.datasets["test"].collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument(
            "--data_dir",
            type=str,
            default="data/multiwoz/processed/zh/",
            help="Path of the dataset.",
        )
        parent_parser.add_argument("--batch_size", type=int, default=2)
        parent_parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="Number of workers in data loader",
        )
        parent_parser.add_argument(
            "--max_len",
            type=int,
            default=512,
        )
        return parent_parser


def build_input_from_segments(
    history: str,
    belief: str,
    tokenizer: BertTokenizer,
) -> Dict[str, List[int]]:
    def tokenize_to_ids(x: str) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))

    bos, eos, bob = tokenizer.convert_tokens_to_ids([BOS, EOS, BELIEF])
    input_ids = [bos, *tokenize_to_ids(history), bob]
    labels = [IGNORE_INDEX] * len(input_ids)
    if belief.strip():
        belief: List[int] = tokenize_to_ids(belief)
        input_ids += [*belief, eos]
        labels += [*belief, eos]

    attention_mask = [1] * len(input_ids)
    assert len(input_ids) == len(labels) == len(attention_mask)
    instance = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    return instance


def pad_truncate_sequence(
    seq: List[List[int]], padding_value: int, max_length: int = 512
) -> torch.LongTensor:
    padded_seq = [
        [padding_value] * (max_length - len(s)) + s[max(0, len(s) - max_length) :]
        for s in seq
    ]
    padded_tensor = torch.tensor(padded_seq, dtype=torch.long)
    return padded_tensor


def pad_truncate_attention_mask(
    seq: List[List[int]], max_length: int = 512
) -> torch.LongTensor:
    attention_mask = [
        [0] * (max_length - len(s)) + s[max(0, len(s) - max_length) :] for s in seq
    ]
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    return attention_mask


def main():
    args = parse_args()
    seed_everything(args.seed)

    tb_logger = loggers.TensorBoardLogger("logs/")
    wandb_logger = loggers.WandbLogger(save_dir="logs/", project="xldst")
    assert wandb_logger.experiment.id
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(
            "ckpts", wandb_logger.experiment.id, "{epoch}-{val_loss:.4f}"
        ),
        save_last=True,
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = Trainer.from_argparse_args(
        args, logger=[tb_logger, wandb_logger], checkpoint_callback=checkpoint_callback
    )
    dm = MultiWOZDataModule(args)
    dm.prepare_data()

    dm.setup("fit")
    model = ConditionalLM(args)
    trainer.fit(model, datamodule=dm)

    dm.setup("test")
    trainer.test(datamodule=dm)


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=13)

    parser = ConditionalLM.add_model_specific_args(parser)
    parser = MultiWOZDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

import logging
import os
from typing import Dict, List, Tuple
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import MBartTokenizer, MBartForConditionalGeneration, AdamW
from transformers.modeling_outputs import Seq2SeqLMOutput

from utils import (
    load_json,
    get_history_utterances,
    build_history_from_utterances,
    stringarize_belief,
    pad_back_or_truncate_start_sequence,
    LANG_CODE,
    IGNORE_INDEX,
)


logging.basicConfig(level=logging.INFO)


class MBartDST(LightningModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = MBartTokenizer.from_pretrained(self.hparams.model_checkpoint)
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.hparams.model_checkpoint
        )

    def forward(self, *args, **kwargs) -> Seq2SeqLMOutput:
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
            default="facebook/mbart-large-cc25",
            help="Dir path to pretrained model",
        )
        parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
        return parser


class CldstMBartDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Tuple[str, str, Dict]]],
        tokenizer: MBartTokenizer,
        lang: str,
        max_source_len: int,
        max_target_len: int,
        num_history_turns: int,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.lang = lang
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.num_history_turns = num_history_turns

        self.data = {
            k: v
            for k, v in self.data.items()
            if k.startswith(self.lang) or self.lang == "both"
        }
        self.turn_ids: List[Tuple[str, int]] = [
            (dialogue_id, turn_id)
            for dialogue_id, turns in self.data.items()
            for turn_id in range(len(turns))
        ]

    def __len__(self) -> int:
        return len(self.turn_ids)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        dialogue_id, turn_id = self.turn_ids[index]
        turns = self.data[dialogue_id]

        turn_lang = dialogue_id.split("_")[0]
        source_lang_code = LANG_CODE[turn_lang]
        target_lang_code = source_lang_code

        _, _, belief = turns[turn_id]
        belief_str = stringarize_belief(belief, add_begin_of_belief=False)
        system_utterances, user_utterances = get_history_utterances(
            turns, self.num_history_turns
        )
        history = build_history_from_utterances(
            system_utterances, user_utterances, turn_lang
        )

        source = history + self.tokenizer.eos_token + source_lang_code
        target = target_lang_code + belief_str + self.tokenizer.eos_token

        source_ids = self.tokenizer.encode(source, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        return {"source_ids": source_ids, "target_ids": target_ids}

    def collate_fn(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_back_or_truncate_start_sequence(
            [i["source_ids"] for i in batch],
            self.tokenizer.pad_token_id,
            self.max_source_len,
        )
        attention_mask = pad_back_or_truncate_start_sequence(
            [[1] * len(i["source_ids"]) for i in batch], 0, self.max_source_len
        )

        target_ids = pad_back_or_truncate_start_sequence(
            [i["target_ids"] for i in batch],
            self.tokenizer.pad_token_id,
            512,  # to avoid truncate start
        )
        target_ids = target_ids[:, : self.max_target_len]  # truncate back
        decoder_input_ids = target_ids[:, :-1].contiguous()
        labels = target_ids[:, 1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return out


class CldstMBartDataModule(LightningDataModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = MBartTokenizer.from_pretrained(self.hparams.model_checkpoint)
        data: Dict[str, Dict[str, List[Dict[str, Dict[str, str]]]]] = load_json(
            self.hparams.data
        )
        if self.hparams.dataset == "both":
            self.datasets: Dict[str, CldstMBartDataset] = {
                dialogue_id: turns
                for dset_name, dset in data.items()
                for dialogue_id, turns in dset.items()
            }
        else:
            self.datasets: Dict[str, CldstMBartDataset] = data[self.hparams.dataset]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            for split in ["train", "val"]:
                self.datasets[split] = CldstMBartDataset(
                    self.datasets[split],
                    self.tokenizer,
                    self.hparams.lang if split == "train" else self.hparams.val_lang,
                    self.hparams.max_source_len,
                    self.hparams.max_target_len,
                    self.hparams.num_history_turns,
                )

        if stage == "test" or stage is None:
            self.datasets["test"] = CldstMBartDataset(
                self.datasets["test"],
                self.tokenizer,
                self.hparams.val_lang,
                self.hparams.max_source_len,
                self.hparams.max_target_len,
                self.hparams.num_history_turns,
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
            "--data",
            type=str,
            default="data/xldst.json",
            help="Path of the dataset.",
        )
        parent_parser.add_argument(
            "--dataset", type=str, choices=["crosswoz", "multiwoz"], required=True
        )
        parent_parser.add_argument(
            "--lang", type=str, choices=["en", "zh", "both"], required=True
        )
        parent_parser.add_argument(
            "--val_lang", type=str, choices=["en", "zh", "both"], default=None
        )
        parent_parser.add_argument("--batch_size", type=int, default=2)
        parent_parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="Number of workers in data loader",
        )
        parent_parser.add_argument(
            "--max_source_len",
            type=int,
            default=512,
        )
        parent_parser.add_argument(
            "--max_target_len",
            type=int,
            default=256,
        )
        parent_parser.add_argument(
            "--num_history_turns",
            type=int,
            help="-1 for all, or positive interger",
            required=True,
        )
        return parent_parser


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
        verbose=True,
    )
    early_stop_callback = EarlyStopping(patience=2, verbose=True)
    trainer = Trainer.from_argparse_args(
        args,
        logger=[tb_logger, wandb_logger],
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
    )
    dm = CldstMBartDataModule(args)
    dm.prepare_data()

    dm.setup("fit")
    model = MBartDST(args)
    trainer.fit(model, datamodule=dm)

    dm.setup("test")
    trainer.test(datamodule=dm)


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=13)

    parser = MBartDST.add_model_specific_args(parser)
    parser = CldstMBartDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(accumulate_grad_batches=2, gradient_clip_val=1.0, precision=16)

    args = parser.parse_args()
    if args.val_lang is None:
        if args.lang == "both":
            if args.dataset == "crosswoz":
                args.val_lang = "en"
            elif args.dataset == "multiwoz":
                args.val_lang = "zh"
            else:
                raise ValueError(f"val_lang cannot be set automatically")
        else:
            args.val_lang = args.lang

    return args


if __name__ == "__main__":
    main()

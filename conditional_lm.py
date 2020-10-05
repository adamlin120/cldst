import logging
import os
from typing import Dict, List, Union, Tuple
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
from transformers import AdamW, BertTokenizer, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from utils import (
    IGNORE_INDEX,
    PAD,
    add_special_tokens_,
    build_input_from_segments,
    pad_truncate_sequence,
    load_tokenizer,
    load_json,
)

logging.basicConfig(level=logging.INFO)


class ConditionalLM(LightningModule):
    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__()
        self.hparams = hparams

        self.tokenizer = load_tokenizer(self.hparams.model_checkpoint, False)
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_checkpoint)
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
            default="ytlin/CDial-GPT2_LCCC-base",
            help="Dir path to pretrained model: ytlin/CDial-GPT2_LCCC-base gpt2, gpt2-medium",
        )
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        return parser


class LmDstDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List[Tuple[str, str, Dict]]],
        tokenizer: Union[BertTokenizer, GPT2LMHeadModel],
        lang: str,
        max_len: int,
        num_history_turns: int,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.lang = lang
        self.max_len = max_len
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

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(PAD)

    def __len__(self) -> int:
        return len(self.turn_ids)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        dialogue_id, turn_id = self.turn_ids[index]
        turns = self.data[dialogue_id]

        if isinstance(self.num_history_turns, int):
            system_utterances, user_utterances, _ = zip(*turns)
            if self.num_history_turns > 0:
                system_utterances = system_utterances[-self.num_history_turns :]
                user_utterances = user_utterances[-self.num_history_turns :]
            _, _, belief = turns[turn_id]
        else:
            raise ValueError(
                f"num_history_turns: {self.num_history_turns} should be -1 or positive integer"
            )

        instance = build_input_from_segments(
            self.tokenizer, system_utterances, user_utterances, belief
        )
        return instance

    def collate_fn(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        out = {
            "input_ids": pad_truncate_sequence(
                [i["input_ids"] for i in batch], self.pad_token_id, self.max_len
            ),
            "labels": pad_truncate_sequence(
                [i["labels"] for i in batch], IGNORE_INDEX, self.max_len
            ),
            "attention_mask": pad_truncate_sequence(
                [[1] * len(i["input_ids"]) for i in batch], 0, self.max_len
            ),
        }
        return out


class LmDstDataModule(LightningDataModule):
    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__()
        self.hparams = hparams

        self.tokenizer = load_tokenizer(
            self.hparams.model_checkpoint, add_special_token=True
        )
        data = load_json(self.hparams.data)
        self.datasets: Dict[str, LmDstDataset] = data[self.hparams.dataset]

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            for split in ["train", "val"]:
                self.datasets[split] = LmDstDataset(
                    self.datasets[split],
                    self.tokenizer,
                    self.hparams.lang,
                    self.hparams.max_len,
                    self.hparams.num_history_turns,
                )

        if stage == "test" or stage is None:
            self.datasets["test"] = LmDstDataset(
                self.datasets[split],
                self.tokenizer,
                self.hparams.lang,
                self.hparams.max_len,
                self.hparams.num_history_turns,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            collate_fn=self.datasets["train"].collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            collate_fn=self.datasets["val"].collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
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
        parent_parser.add_argument("--batch_size", type=int, default=16)
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
    dm = LmDstDataModule(args)
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
    parser = LmDstDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(accumulate_grad_batches=2, gradient_clip_val=1.0, precision=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

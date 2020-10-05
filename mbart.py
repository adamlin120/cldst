import logging
import os
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import MBartTokenizer, MBartForConditionalGeneration, AdamW
from transformers.modeling_outputs import Seq2SeqLMOutput


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
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        return parser


class CldstMBartDataset(Dataset):
    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path
        self.data: Dict[str, torch.Tensor] = torch.load(self.path.open("rb"))

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, index: int) -> int:
        return index

    def collate_fn(self, batch_of_index: List[int]) -> Dict[str, torch.Tensor]:
        return {k: v[batch_of_index] for k, v in self.data.items()}


class CldstMBartDataModule(LightningDataModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = MBartTokenizer.from_pretrained(self.hparams.model_checkpoint)
        self.datasets: Dict[str, CldstMBartDataset] = {}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            for split in ["train", "val"]:
                self.datasets[split] = CldstMBartDataset(
                    Path(self.hparams.data_dir) / f"{split}.pt"
                )

        if stage == "test" or stage is None:
            self.datasets["test"] = CldstMBartDataset(
                Path(self.hparams.data_dir) / "test.pt"
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
        parent_parser.add_argument("--batch_size", type=int, default=16)
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
    return args


if __name__ == "__main__":
    main()

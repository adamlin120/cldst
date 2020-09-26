import json
from typing import Dict, List
from pathlib import Path
from argparse import ArgumentParser, Namespace

import ipdb
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from module import BOS, EOS, BELIEF, IGNORE_INDEX


class MultiwozDataset(Dataset):
    def __init__(self, path: Path, tokenizer: BertTokenizer) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.data = json.loads(self.path.read_text())
        self.turn_ids: List[str] = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: str) -> Dict[str, List[int]]:
        turn_id = self.turn_ids[index]
        turn = self.data[turn_id]
        instance = build_input_from_segments(
            turn["history"], turn.get("belief", ""), self.tokenizer
        )
        return instance

    @staticmethod
    def collate_fn():
        ipdb.set_trace()
        pass


class MultiWOZDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.tokenizer_name)
        self.datasets: Dict[str, Dataset] = {}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            for split in ["train", "val"]:
                self.datasets[split] = MultiwozDataset(
                    Path(self.hparams.data_dir) / f"{split}.json", self.tokenizer
                )

        if stage == "test" or stage is None:
            self.datasets["test"] = MultiwozDataset(
                Path(self.hparams.data_dir) / "test.json", self.tokenizer
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument(
            "--data_dir",
            type=str,
            default="data/multiwoz/seq2seq/zh/",
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
            "--tokenizer_name", type=str, default="models/mega-bert-tok/"
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
    belief: List[int] = tokenize_to_ids(belief)
    input_ids = [bos, *tokenize_to_ids(history), bob]
    labels = [IGNORE_INDEX] * len(input_ids)
    input_ids += [*belief, eos]
    labels += [*belief, eos]
    ipdb.set_trace()

    assert len(input_ids) == len(labels)
    instance = {"input_ids": input_ids, "labels": labels}
    return instance

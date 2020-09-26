import json
from typing import Dict, List
from pathlib import Path
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from module import BOS, EOS, BELIEF, IGNORE_INDEX, ATTR_TO_SPECIAL_TOKEN, PAD


class MultiwozDataset(Dataset):
    def __init__(
        self, path: Path, tokenizer: BertTokenizer, max_len: int = 256
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data = json.loads(self.path.read_text())
        self.turn_ids: List[str] = list(self.data.keys())
        self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(PAD)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: str) -> Dict[str, List[int]]:
        turn_id = self.turn_ids[index]
        turn = self.data[turn_id]
        instance = build_input_from_segments(
            turn["history"], turn.get("belief", ""), self.tokenizer
        )
        return instance

    def collate_fn(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        return {
            tensor_name: pad_truncate_sequence(
                [i[tensor_name] for i in batch], self.pad_token_id, self.max_len
            )
            for tensor_name in batch[0].keys()
        }


class MultiWOZDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(
            self.hparams.model_checkpoint, do_lower_case=True
        )
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
            "--max_len",
            type=int,
            default=256,
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

    assert len(input_ids) == len(labels)
    instance = {"input_ids": input_ids, "labels": labels}
    return instance


def build_test_string(history: str) -> str:
    return f"{BOS} {history} {BELIEF}"


def pad_truncate_sequence(
    seq: List[List[int]], padding_value: int, max_length: int = 1024
) -> torch.LongTensor:
    max_length = min(max_length, max(len(s) for s in seq))
    padded_seq = [
        s[max(0, len(s) - max_length) :] + [padding_value] * (max_length - len(s))
        for s in seq
    ]
    padded_tensor = torch.tensor(padded_seq, dtype=torch.long)
    return padded_tensor

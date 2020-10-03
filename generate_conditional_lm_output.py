import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict

import ipdb
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm.auto import tqdm

from conditional_lm import EOS, MultiwozDataset, PAD

MAX_LENGTH = 512
MAX_FOR_PROMPT = MAX_LENGTH - 100


logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    device = torch.device(
        "cpu"
        if args.cuda_device < 0 or not torch.cuda.is_available()
        else args.cuda_device
    )
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    eos_token_id = tokenizer.convert_tokens_to_ids(EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(PAD)

    test_set = json.loads(args.test_set.read_text())

    dataset = MultiwozDataset(args.test_set, tokenizer, 512)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn
    )
    preds = []
    for batch in tqdm(loader):
        gen = model.generate(
            batch["input_ids"],
            max_length=MAX_LENGTH,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        gen_str = tokenizer.batch_decode(gen)
        preds.extend(gen_str)
        ipdb.set_trace()

    pred_dump = defaultdict(list)
    for pred, id in zip(preds, test_set.keys()):
        dialogue_id, turn_id = id.split("-")
        pred_dump[dialogue_id].append(pred)

    Path(args.test_set + "." + args.output_tag).write_text(
        json.dumps(pred_dump, ensure_ascii=False, indent=4)
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_tag")
    parser.add_argument("test_set", type=Path)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

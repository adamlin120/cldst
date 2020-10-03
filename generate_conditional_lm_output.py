import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm.auto import tqdm

from conditional_lm import EOS, MultiwozDataset, PAD

MAX_LENGTH = 512
MAX_FOR_PROMPT = MAX_LENGTH - 128


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

    test_set_path = (
        Path("./data") / args.dataset / "processed" / args.lang / f"{args.split}.json"
    )
    test_set = json.loads(test_set_path.read_text())

    dataset = MultiwozDataset(test_set_path, tokenizer, MAX_FOR_PROMPT, True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=cpu_count(),
    )
    preds = []
    for batch in tqdm(loader):
        gen = model.generate(
            batch["input_ids"].to(device),
            max_length=MAX_LENGTH,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        gen_str = tokenizer.batch_decode(gen)
        preds.extend(gen_str)

    pred_dump = defaultdict(list)
    for pred, id in zip(preds, test_set.keys()):
        dialogue_id, turn_id = id.split("-")
        pred_dump[dialogue_id].append(pred)

    normalized_ckpt = args.checkpoint_path.replace("/", "_")
    output_path = (
        Path("./submission")
        / args.dataset
        / args.lang
        / args.split
        / f"{normalized_ckpt}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pred_dump, ensure_ascii=False, indent=4))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("dataset", type=str, choices=["multiwoz", "crosswoz"])
    parser.add_argument("lang", type=str, choices=["en", "zh"])
    parser.add_argument("split", type=str, choices=["val", "test", "test-250"])
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, BertTokenizer
from tqdm.auto import tqdm

from conditional_lm import EOS, PAD, build_input_from_segments

MAX_LENGTH = 512
MIN_BELIEF_LEN = 128
MAX_FOR_PROMPT = MAX_LENGTH - MIN_BELIEF_LEN


logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    device = torch.device(
        "cpu"
        if args.cuda_device < 0 or not torch.cuda.is_available()
        else args.cuda_device
    )
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).eval().to(device)

    tokenzier_class = AutoTokenizer if not args.use_bert_tokenizer else BertTokenizer
    tokenizer = tokenzier_class.from_pretrained(args.checkpoint_path)
    eos_token_id = tokenizer.convert_tokens_to_ids(EOS)
    pad_token_id = tokenizer.convert_tokens_to_ids(PAD)

    test_set_path = (
        Path("./data") / args.dataset / "processed" / args.lang / f"{args.split}.json"
    )
    test_set = json.loads(test_set_path.read_text())

    preds = []
    for i, (id, turn) in tqdm(enumerate(test_set.items()), total=len(test_set)):
        input_ids = build_input_from_segments(turn["history"], None, tokenizer)[
            "input_ids"
        ]
        if len(input_ids) > MAX_FOR_PROMPT:
            input_ids = input_ids[len(input_ids) - MAX_FOR_PROMPT :]
        assert len(input_ids) <= MAX_FOR_PROMPT
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        gen = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        gen_str = tokenizer.batch_decode(gen)
        preds.extend(gen_str)
        if i > 4 and args.debug:
            break

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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_bert_tokenizer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

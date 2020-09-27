import json
import os
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from transformers import BertTokenizer, GPT2LMHeadModel
from tqdm.auto import tqdm

from module import EOS
from multiwoz_data_module import build_test_string, build_input_from_segments


logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).eval()
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint_path)

    eos_token_id = tokenizer.convert_tokens_to_ids(EOS)

    test_set = json.loads(Path(args.test_set).read_text())
    pred = {}

    for i, (id, turn) in tqdm(enumerate(test_set.items())):
        history = turn["history"]
        history = build_test_string(history)
        input_ids = tokenizer(history, add_special_tokens=False)
        gen = model.generate(
            input_ids,
            max_length=512,
            eos_token_id=eos_token_id,
            clean_up_tokenization_spaces=False,
        )
        pred[id] = tokenizer.decode(gen[0])
        if i > 10:
            break
    Path(
        args.test_set + "." + os.path.basename(os.path.dirname(args.checkpoint_path))
    ).write_text(json.dumps(pred, ensure_ascii=False, indent=4))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument(
        "--test_set", type=str, default="./data/multiwoz/processed/zh/test.json"
    )
    parser.add_argument("--cuda_device", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

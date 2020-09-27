import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint as print
import pdb

from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

from module import EOS
from multiwoz_data_module import build_test_string


logging.basicConfig(level=logging.INFO)


def main(args: Namespace):
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).eval()
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint_path)

    eos_token_id = tokenizer.convert_tokens_to_ids(EOS)

    pipeline = TextGenerationPipeline(
        model, tokenizer, framework="pt", device=args.cuda_device
    )

    test_set = json.loads(Path(args.test_set).read_text())

    for id, turn in test_set.items():
        # history = input("History:")
        history = turn["history"]
        history = build_test_string(history)
        gen = pipeline(
            history,
            max_length=512,
            eos_token_id=eos_token_id,
            clean_up_tokenization_spaces=False,
            return_tensors=True,
        )
        print(gen, turn["belief"])
        pdb.set_trace()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument(
        "--test_set", type=str, default="./data/multiwoz/seq2seq/zh/test.json"
    )
    parser.add_argument("--cuda_device", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

import logging
from argparse import ArgumentParser, Namespace

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

    while True:
        history = input("History:")
        history = build_test_string(history)
        gen = pipeline(
            history,
            max_length=512,
            eos_token_id=eos_token_id,
            # top_k=50,
            # top_p=0.95,
        )
        print(gen)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--cuda_device", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

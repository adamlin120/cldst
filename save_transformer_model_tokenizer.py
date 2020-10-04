import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from conditional_lm import ConditionalLM
from mbart import MBartDST

logging.basicConfig(level=logging.INFO)


def main():
    args = parse_args()
    module_class = MBartDST if args.mbart else ConditionalLM
    model = module_class.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        hparams_file=args.hparams_file,
    )
    args.save_dir.mkdir(exist_ok=True, parents=True)
    model.model.save_pretrained(args.save_dir)
    model.tokenizer.save_pretrained(args.save_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("checkpoint_path")
    parser.add_argument("hparams_file")
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--mbart", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

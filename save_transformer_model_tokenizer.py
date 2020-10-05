import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from conditional_lm import ConditionalLM
from mbart import MBartDST

logging.basicConfig(level=logging.INFO)


def main():
    args = parse_args()
    module_class = MBartDST if args.mbart else ConditionalLM
    model = module_class.load_from_checkpoint(checkpoint_path=str(args.checkpoint_path))
    dir = args.checkpoint_path.parent
    save_dir = dir / dir.stem
    save_dir.mkdir(exist_ok=True, parents=True)
    model.model.save_pretrained(save_dir)
    model.tokenizer.save_pretrained(save_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--mbart", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

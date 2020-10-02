import json
import os
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import defaultdict

import torch
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm

from module import EOS
from multiwoz_data_module import build_test_string

MAX_LENGTH = 512
MAX_FOR_PROMPT = MAX_LENGTH - 100

ONTOLOGY = json.loads(Path("./data/multiwoz/zh/ontology-data.json").read_text())


logging.basicConfig(level=logging.INFO)

belief_template = {
    "出租车": {
        "出发时间": "",
        "目的地": "",
        "出发地": "",
        "到达时间": "",
    },
    "餐厅": {
        "时间": "",
        "日期": "",
        "人数": "",
        "食物": "",
        "价格范围": "",
        "名称": "",
        "区域": "",
    },
    "公共汽车": {
        "人数": "",
        "出发时间": "",
        "目的地": "",
        "日期": "",
        "到达时间": "",
        "出发地": "",
    },
    "旅馆": {
        "停留天数": "",
        "日期": "",
        "人数": "",
        "名称": "",
        "区域": "",
        "停车处": "",
        "价格范围": "",
        "星级": "",
        "互联网": "",
        "类型": "",
    },
    "景点": {
        "类型": "",
        "名称": "",
        "区域": "",
    },
    "列车": {
        "票价": "",
        "人数": "",
        "出发时间": "",
        "目的地": "",
        "日期": "",
        "到达时间": "",
        "出发地": "",
    },
}


def main(args: Namespace):
    device = torch.device("cpu" if args.cuda_device < 0 else args.cuda_device)
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path).eval().to(device)

    tokenizer_class = GPT2Tokenizer if args.gpt2_tokenizer else BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.checkpoint_path)

    eos_token_id = tokenizer.convert_tokens_to_ids(EOS)

    test_set = json.loads(Path(args.test_set).read_text())
    pred = defaultdict(list)

    for i, (id, turn) in tqdm(enumerate(test_set.items()), total=len(test_set)):
        dialogue_id = id.split("-", 1)[0]
        history = turn["history"]
        history = build_test_string(history)
        input_ids = tokenizer(history, add_special_tokens=False, return_tensors="pt",)[
            "input_ids"
        ].to(device)
        if len(input_ids[0]) >= MAX_FOR_PROMPT:
            input_ids = input_ids[:, -MAX_FOR_PROMPT:]
        gen = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
        )
        gen_str = tokenizer.decode(gen[0])
        pred[dialogue_id].append(gen_str)

        if args.debug and i > 4:
            break

    Path(args.test_set + "." + args.output_tag).write_text(
        json.dumps(pred, ensure_ascii=False, indent=4)
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_tag")
    parser.add_argument(
        "--test_set", type=str, default="./data/multiwoz/processed/zh/test.json"
    )
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument(
        "--gpt2_tokenizer",
        action="store_true",
        help="use gpt2 tokenizer instead of bert tokenizer",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

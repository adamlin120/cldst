import json
import logging
import re
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm


MAX_LENGTH = 512
MAX_FOR_PROMPT = MAX_LENGTH - 100

NOT_MENSION = "未提及"

logging.basicConfig(level=logging.INFO)

belief_templates = {
    "multiwoz-zh": {
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
    },
    "crosswoz-en": {
        "Attraction": {
            "name": "",
            "fee": "",
            "duration": "",
            "rating": "",
            "nearby attract.": "",
            "nearby rest.": "",
            "nearby hotels": "",
        },
        "Restaurant": {
            "name": "",
            "dishes": "",
            "cost": "",
            "rating": "",
            "nearby attract.": "",
            "nearby rest.": "",
            "nearby hotels": "",
        },
        "Hotel": {
            "name": "",
            "type": "",
            "Hotel Facilities": "",
            "price": "",
            "rating": "",
            "nearby attract.": "",
            "nearby rest.": "",
            "nearby hotels": "",
        },
        "Metro": {"from": "", "to": ""},
        "Taxi": {"from": "", "to": ""},
    },
}
SLOTS_NEED_NOT_MENSIONED = {
    "出租车": {"出发时间", "目的地", "出发地", "到达时间"},
    "餐厅": {"食物", "价格范围", "名称", "区域"},
    "公共汽车": {"人数", "出发时间", "目的地", "日期", "到达时间", "出发地"},
    "旅馆": {"名称", "区域", "停车处", "价格范围", "星级", "互联网", "类型"},
    "景点": {"类型", "名称", "区域"},
    "列车": {"出发时间", "目的地", "日期", "到达时间", "出发地"},
}


def add_not_mension(belief: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {
        domain: {
            slot_name: NOT_MENSION
            if any(v.strip() for v in domain_slots.values())
            and slot_name in SLOTS_NEED_NOT_MENSIONED.get(domain, {})
            and not slot_value.strip()
            else slot_value
            for slot_name, slot_value in domain_slots.items()
        }
        for domain, domain_slots in belief.items()
    }


def parse_belief(gen: str, belief_template, ontology) -> Dict[str, Dict[str, str]]:
    slots = list(
        map(
            lambda x: re.split("<SLOT_NAME>|<SLOT_VALUE>", x),
            re.split(
                "<SLOT>",
                gen.split("<BELIEF>", 1)[-1].split("<eos>", 1)[0].strip(),
            )[1:],
        )
    )
    belief = deepcopy(belief_template)
    for slot in slots:
        domain = slot[0]
        slot_name = slot[1] if len(slot) >= 2 else ""
        slot_value = slot[2] if len(slot) >= 3 else ""

        domain = remove_spaces(domain)
        if not domain.strip():
            continue
        if domain not in belief.keys():
            domain = get_close_matches(domain, list(belief.keys()), 1, 0)[0]

        slot_name = remove_spaces(slot_name)
        if not slot_name.strip():
            continue
        possible_slot_names = belief[domain].keys()
        if slot_name not in possible_slot_names:
            slot_name = get_close_matches(slot_name, possible_slot_names, 1, 0)[0]

        slot_value = remove_spaces(slot_value)
        if not slot_value.split():
            continue
        possible_values = ontology[domain][slot_name]
        if slot_value not in possible_values:
            slot_value = get_close_matches(slot_value, possible_values, 1, 0)[0]

        belief[domain][slot_name] = slot_value
    return belief


def remove_spaces(text: str) -> str:
    return "".join(text.split())


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("dataset", type=str, choices=["multiwoz", "crosswoz"])
    parser.add_argument("lang", type=str, choices=["en", "zh"])
    return parser.parse_args()


def main():
    args = parse_args()
    ontology = json.loads(
        Path(f"./data/{args.dataset}/{args.lang}/ontology-data.json").read_text()
    )
    test_set = json.loads(args.input.read_text())

    belief_template = belief_templates[f"{args.dataset}-{args.lang}"]

    for dialogue_id, preds in tqdm(test_set.items()):
        for i, gen_str in enumerate(preds):
            belief = parse_belief(gen_str, belief_template, ontology)
            belief = add_not_mension(belief)
            test_set[dialogue_id][i] = belief

    args.output.write_text(json.dumps(test_set, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from itertools import cycle
from typing import Dict, List

from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


SLOT_SEP = "<SLOT>"
SLOT_NAME_SEP = "<SLOT_NAME>"
SLOT_VALUE_SEP = "<SLOT_VALUE>"

USER = {"en": "user", "zh": "用戶"}
SYSTEM = {"en": "system", "zh": "系统"}

NOT_MENTIONED = {"", "未提及", "none", "not mentioned"}

SLOT_LIST = {
    "en": "./data/multiwoz/slot_list_en.json",
    "zh": "./data/multiwoz/slot_list_zh.json",
}
READABLE_SLOT_LIST = "./data/multiwoz/readable_slots.json"

FILENAMES = {
    "en": {
        "train": "train_Cleaned.json",
        "val": "val_Cleaned.json",
        "test": "test_Cleaned.json",
    },
    "zh": {
        "train": "train.json",
        "val": "val.json",
        "val_human": "human_val.json",
        "test": "test.json",
        "test-250": "data.json",
    },
}


def canonicalize_slot_name(belief: Dict, mode: str) -> Dict:
    canonicalized_belief = {}
    for domain, origin_slots in belief.items():
        for slot_type, slots in origin_slots.items():
            for slot_origin_name, slot_value in slots.items():
                if slot_type == "book" and mode == "en":
                    if slot_origin_name == "booked":
                        continue
                    slot_name = f"{domain}-book{slot_origin_name}"
                else:
                    slot_name = f"{domain}-{slot_origin_name}"
                canonicalized_belief[slot_name] = slot_value
    return canonicalized_belief


def clean_slot_value(
    belief: Dict,
    sorted_slot_list: List[str],
    readable_slot: Dict[str, str],
    split_character: bool,
) -> List[str]:
    clean_belief: List[str] = []
    for slot_name in sorted_slot_list:
        if is_empty_slot(belief.get(slot_name, "")):
            continue
        domain, readable_slot_name = readable_slot.get(slot_name, slot_name).split("-")
        slot_value = belief[slot_name]
        if split_character:
            domain = " ".join(domain)
            readable_slot_name = " ".join(readable_slot_name)
            slot_value = " ".join(slot_value)
        clean_belief.append(
            f"{SLOT_SEP} {domain} {SLOT_NAME_SEP} {readable_slot_name} {SLOT_VALUE_SEP} {slot_value}"
        )
    return clean_belief


def split_chinese_characters_in_history(history: str) -> str:
    return " ".join(" ".join(history).split()).strip()


def is_empty_slot(slot_value: str) -> bool:
    slot_value = slot_value.lower().strip()
    return slot_value in NOT_MENTIONED


def main():
    args = parse_args()
    sorted_slot_list: List[str] = sorted(
        json.loads(Path(SLOT_LIST[args.mode]).read_text())
    )
    readable_slots: Dict[str, str] = json.loads(Path(READABLE_SLOT_LIST).read_text())

    user, system = USER[args.mode], SYSTEM[args.mode]

    for split, filename in FILENAMES[args.mode].items():
        data = {}
        dataset_dir = (
            args.multiwoz_en_dir if args.mode == "en" else args.multiwoz_zn_dir
        )
        filepath = dataset_dir / filename
        for dialogue_id, item in tqdm(
            json.loads(filepath.read_text()).items(), desc=split
        ):
            history = ""
            for turn_id, (speaker, turn) in enumerate(
                zip(cycle([user, system]), item["log"])
            ):
                if speaker == user:
                    history += f"{speaker} : {turn['text']} "
                elif speaker == system:
                    belief = turn.get("metadata", {})
                    belief = canonicalize_slot_name(belief, args.mode)
                    belief = clean_slot_value(
                        belief, sorted_slot_list, readable_slots, args.mode == "zh"
                    )
                    belief = " ".join(belief)

                    data[f"{dialogue_id}-{turn_id}"] = {
                        "belief": belief,
                        "history": history.strip()
                        if args.mode == "en"
                        else split_chinese_characters_in_history(history),
                    }
                    history += f"{speaker} : {turn['text']} "
                else:
                    raise ValueError(f"Unexpected speaker: {speaker}")

        logging.info(f"Number of instance in in split {split.upper()}: {len(data):,}")
        output_dir = args.output_dir / args.mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(f"{str(output_dir / split)}.json")
        logging.info(f"Writing {split.upper()} dataset to {output_path}")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("mode", type=str, choices=["en", "zh"])
    parser.add_argument("--multiwoz_en_dir", type=Path, default="./data/multiwoz/en/")
    parser.add_argument("--multiwoz_zn_dir", type=Path, default="./data/multiwoz/zh/")
    parser.add_argument("--output_dir", type=Path, default="./data/multiwoz/processed/")
    return parser.parse_args()


if __name__ == "__main__":
    main()

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List
from collections import namedtuple, defaultdict

from tqdm.auto import tqdm

from preprocess_multiwoz_data import (
    USER,
    SYSTEM,
    split_chinese_characters_in_history,
    clean_slot_value,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


SLOT_LIST = {
    "en": "./data/crosswoz/slot_list_en.json",
    "zh": "./data/crosswoz/slot_list_zh.json",
}
READABLE_SLOT_LIST = "./data/crosswoz/readable_slots.json"
FILENAMES = {
    "en": {
        "train": "train.json",
        "val": "val.json",
        "val_human": "human_val.json",
        "test": "test.json",
        "test-250": "data.json",
    },
    "zh": {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    },
}

Turn = namedtuple("turn", ["sys", "user", "belief"])


def extract_cross_woz(raw_data) -> Dict[str, List[Turn]]:
    data = defaultdict(list)
    for dialog_id, dialog in raw_data.items():
        turns = dialog["messages"]
        for i in range(0, len(turns), 2):
            sys_utt = turns[i - 1]["content"] if i else ""
            user_utt = turns[i]["content"]
            state = {}
            for domain_name, domain in turns[i + 1].get("sys_state_init", {}).items():
                for slot_name, value in domain.items():
                    if slot_name == "selectedResults":
                        continue
                    state[f"{domain_name}-{slot_name}"] = value
            data[f"{dialog_id}"].append(Turn(sys_utt, user_utt, state))
    return data


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
            args.crosswoz_en_dir if args.mode == "en" else args.crosswoz_zn_dir
        )
        filepath = dataset_dir / filename
        raw_data = json.loads(filepath.read_text())
        dialogues = extract_cross_woz(raw_data)
        for dialogue_id, dialogue in tqdm(dialogues.items(), desc=split):
            history = ""
            for turn_id, turn in enumerate(dialogue):
                belief = clean_slot_value(
                    turn.belief, sorted_slot_list, readable_slots, args.mode == "zh"
                )
                belief = " ".join(belief)

                if turn.sys.strip():
                    history += f" {system} : {turn.sys} "
                if turn.user.strip():
                    history += f" {user} : {turn.user} "

                data[f"{dialogue_id}-{turn_id}"] = {
                    "belief": belief,
                    "history": history.strip()
                    if args.mode == "en"
                    else split_chinese_characters_in_history(history),
                }

        logging.info(f"Number of instance in in split {split.upper()}: {len(data):,}")
        output_dir = args.output_dir / args.mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(f"{str(output_dir / split)}.json")
        logging.info(f"Writing {split.upper()} dataset to {output_path}")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("mode", type=str, choices=["en", "zh"])
    parser.add_argument("--crosswoz_zn_dir", type=Path, default="./data/crosswoz/zh/")
    parser.add_argument("--crosswoz_en_dir", type=Path, default="./data/crosswoz/en/")
    parser.add_argument("--output_dir", type=Path, default="./data/crosswoz/processed/")
    return parser.parse_args()


if __name__ == "__main__":
    main()

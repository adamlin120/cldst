import json
from argparse import ArgumentParser
from pathlib import Path
from itertools import cycle
from typing import Dict, List
from difflib import get_close_matches

from tqdm.auto import tqdm

USER = "user"
SYSTEM = "system"

MC_SLOT_LIST = "./categorical.json"
MC_POSSIBLE_ANSWERS = "./possible_values.json"
SLOT_QUESTION = "./question.json"

FILENAMES = {
    "train": "train_Cleaned.json",
    "val": "val_Cleaned.json",
    "test": "test_Cleaned.json",
}


def canonicalize_slot_name(belief: Dict) -> Dict:
    canonicalized_belief = {}
    for domain, origin_slots in belief.items():
        for slot_type, slots in origin_slots.items():
            for slot_origin_name, slot_value in slots.items():
                if slot_type == "book":
                    if slot_origin_name == "booked":
                        continue
                    slot_name = f"{domain}-book{slot_origin_name}"
                elif slot_type == "semi":
                    slot_name = f"{domain}-{slot_origin_name}"
                else:
                    raise Exception(f"Unknown slot type: {slot_type}")
                canonicalized_belief[slot_name] = slot_value
    return canonicalized_belief


def get_mc_slots(belief: Dict, mc_slots: List[str]) -> Dict:
    return {slot: belief.get(slot, "") for slot in mc_slots}


def clean_slot_value(belief: Dict) -> Dict:
    clean_belief = {}
    for slot_name, slot_value in belief.items():
        if slot_value.lower() == "none" or not slot_value.strip():
            new_value = "not mentioned"
        else:
            new_value = slot_value
        clean_belief[slot_name] = new_value
    return clean_belief


def get_answer(belief: Dict, possible_answer: Dict[str, List]) -> Dict:
    new_belief = {}
    for slot_name, slot_value in belief.items():
        options = possible_answer[slot_name]
        if slot_value not in options:
            slot_value = get_close_matches(slot_value, options, 1, 0)[0]
        new_belief[slot_name] = slot_value
    return new_belief


def main():
    args = parse_args()
    mc_slots = json.loads(Path(MC_SLOT_LIST).read_text())
    possible_answer = json.loads(Path(MC_POSSIBLE_ANSWERS).read_text())
    questions = json.loads(Path(SLOT_QUESTION).read_text())

    for split, filename in FILENAMES.items():
        mc_data = {
            slot_name: {
                "possible_answer": answer,
                "question": questions[slot_name],
                "turns": [],
            }
            for slot_name, answer in possible_answer.items()
        }
        filepath = args.multiwoz21_clean_dir / filename
        for dialogue_id, item in tqdm(json.loads(filepath.read_text()).items()):
            history = ""
            for turn_id, (speaker, turn) in enumerate(
                zip(cycle([USER, SYSTEM]), item["log"])
            ):
                if speaker == USER:
                    history += f" {speaker} : {turn['text']} "
                elif speaker == SYSTEM:
                    belief = turn["metadata"]
                    belief = canonicalize_slot_name(belief)
                    belief = get_mc_slots(belief, mc_slots)
                    belief = clean_slot_value(belief)
                    belief = get_answer(belief, possible_answer)
                    for slot_name, slot_value in belief.items():
                        mc_data[slot_name]["turns"].append(
                            {
                                "id": f"{dialogue_id}-{turn_id}",
                                "history": history.strip(),
                                "label": slot_value,
                            }
                        )
                    history += f" {speaker} : {turn['text']} "
                else:
                    raise ValueError(f"Unexpected speaker: {speaker}")

        for slot_name, slot_data in mc_data.items():
            slot_output_dir = args.output_dir / slot_name
            slot_output_dir.mkdir(parents=True, exist_ok=True)
            Path(f"{str(slot_output_dir / split)}.json").write_text(
                json.dumps(slot_data, indent=2)
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--multiwoz21_clean_dir", type=Path, default="./MultiWOZ2.1_Cleaned/"
    )
    parser.add_argument("--output_dir", type=Path, default="mc/21_clean/")
    return parser.parse_args()


if __name__ == "__main__":
    main()

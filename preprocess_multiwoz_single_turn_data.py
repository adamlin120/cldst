import json
import logging
from pathlib import Path
from itertools import cycle
from typing import Dict, List
from copy import deepcopy

from tqdm.auto import tqdm

from preprocess_multiwoz_data import (
    SLOT_VALUE_SEP,
    USER,
    SYSTEM,
    SLOT_LIST,
    READABLE_SLOT_LIST,
    FILENAMES,
    canonicalize_slot_name,
    clean_slot_value,
    split_chinese_characters_in_history,
    parse_args,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DELETE = {"en": "delete", "zh": "刪 除"}


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
            history = []
            prev_belief_str_list = []
            for turn_id, (speaker, turn) in enumerate(
                zip(cycle([user, system]), item["log"])
            ):
                if speaker == user:
                    history.append(f"{speaker} : {turn['text']}")
                elif speaker == system:
                    belief = turn.get("metadata", {})
                    belief = canonicalize_slot_name(belief, args.mode)
                    belief_str_list = clean_slot_value(
                        belief, sorted_slot_list, readable_slots, args.mode == "zh"
                    )

                    new_belief_str_list = deepcopy(belief_str_list)
                    for prev_slot in prev_belief_str_list:
                        if prev_slot in new_belief_str_list:
                            new_belief_str_list.remove(prev_slot)
                        else:
                            domain_slot = prev_slot.split("<SLOT_VALUE>")[0].strip()
                            if all(
                                domain_slot not in new_belief
                                for new_belief in new_belief_str_list
                            ):
                                new_belief_str_list.append(
                                    f"{domain_slot} {SLOT_VALUE_SEP} {DELETE[args.mode]}"
                                )
                    new_belief_str = " ".join(new_belief_str_list)
                    prev_belief_str = " ".join(prev_belief_str_list)

                    last_four_utterance = " ".join(history[-4:])

                    data[f"{dialogue_id}-{turn_id}"] = {
                        "belief": new_belief_str,
                        "history": (
                            last_four_utterance
                            if args.mode == "en"
                            else split_chinese_characters_in_history(
                                last_four_utterance
                            )
                        )
                        + f" {prev_belief_str}",
                    }
                    prev_belief_str_list = belief_str_list
                    history.append(f"{speaker} : {turn['text']}")
                else:
                    raise ValueError(f"Unexpected speaker: {speaker}")

        logging.info(f"Number of instance in in split {split.upper()}: {len(data):,}")
        output_dir = args.output_dir / f"{args.mode}_single_turn"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(f"{str(output_dir / split)}.json")
        logging.info(f"Writing {split.upper()} dataset to {output_path}")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

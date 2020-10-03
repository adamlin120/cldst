import json
import logging
from pathlib import Path
from typing import Dict, List
from copy import deepcopy

from tqdm.auto import tqdm

from preprocess_multiwoz_data import (
    USER,
    SYSTEM,
    split_chinese_characters_in_history,
    clean_slot_value,
    SLOT_VALUE_SEP,
)
from preprocess_crosswoz_data import (
    SLOT_LIST,
    READABLE_SLOT_LIST,
    FILENAMES,
    extract_cross_woz,
    parse_args,
)
from preprocess_multiwoz_single_turn_data import DELETE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


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
            history = []
            prev_belief_list = []
            for turn_id, turn in enumerate(dialogue):
                belief_list = clean_slot_value(
                    turn.belief, sorted_slot_list, readable_slots, args.mode == "zh"
                )

                new_belief_list = deepcopy(belief_list)
                for prev_slot in prev_belief_list:
                    if prev_slot in new_belief_list:
                        new_belief_list.remove(prev_slot)
                    else:
                        domain_slot = prev_slot.split("<SLOT_VALUE>")[0].strip()
                        if all(
                            domain_slot not in new_belief
                            for new_belief in new_belief_list
                        ):
                            new_belief_list.append(
                                f"{domain_slot} {SLOT_VALUE_SEP} {DELETE[args.mode]}"
                            )
                new_belief_str = " ".join(new_belief_list)
                prev_belief_str = " ".join(prev_belief_list)

                if turn.sys.strip():
                    history.append(f"{system} : {turn.sys}")
                if turn.user.strip():
                    history.append(f"{user} : {turn.user}")
                history_str = " ".join(history[-4:]).strip()

                data[f"{dialogue_id}-{turn_id}"] = {
                    "belief": new_belief_str,
                    "history": (
                        history_str
                        if args.mode == "en"
                        else split_chinese_characters_in_history(history_str)
                    )
                    + f" {prev_belief_str}",
                }
                prev_belief_list = belief_list

        logging.info(f"Number of instance in in split {split.upper()}: {len(data):,}")
        output_dir = args.output_dir / f"{args.mode}_single_turn"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(f"{str(output_dir / split)}.json")
        logging.info(f"Writing {split.upper()} dataset to {output_path}")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

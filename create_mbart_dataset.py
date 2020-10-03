import json
import re
import random
from pathlib import Path

import torch
from transformers import MBartTokenizer
from transformers.modeling_bart import shift_tokens_right
from tqdm.auto import tqdm


random.seed(13)


LANG2LANG_CODE = {"zh": "zh_CN", "en": "en_XX"}
IGNORE_IDX = -100

max_length = 512
max_target_length = 256


def clean_chinese_spaces(text: str) -> str:
    return " ".join(
        [
            "".join(split.split()).strip()
            for split in re.split("(<SLOT>|<SLOT_NAME>|<SLOT_VALUE>)", text)
        ]
    ).strip()


def main():
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    data_dir = Path("./data/")
    for dataset_name in ["multiwoz", "crosswoz"]:
        for split in tqdm(["train", "val", "test"], desc=dataset_name):
            datasets = {
                "input_ids": [],
                "attention_mask": [],
                "decoder_input_ids": [],
                "labels": [],
            }
            for lang in ["en_single_turn", "zh_single_turn"]:
                if split != "train":
                    if (dataset_name == "multiwoz" and "zh" not in lang) or (
                        dataset_name == "crosswoz" and "en" not in lang
                    ):
                        continue
                file_path = (
                    data_dir / dataset_name / "processed" / lang / f"{split}.json"
                )
                data = json.loads(file_path.read_text())
                if split == "train" and (
                    (dataset_name == "multiwoz" and "zh" not in lang)
                    or (dataset_name == "crosswoz" and "en" not in lang)
                ):
                    data = dict(random.sample(data.items(), len(data) // 2))
                sources = [v["history"] for v in data.values()]
                targets = [v["belief"] for v in data.values()]
                if "zh" in lang:
                    sources = list(map(clean_chinese_spaces, sources))
                    targets = list(map(clean_chinese_spaces, targets))
                batch = tokenizer.prepare_seq2seq_batch(
                    src_texts=sources,
                    src_lang=LANG2LANG_CODE[lang.split("_")[0]],
                    tgt_texts=targets,
                    tgt_lang=LANG2LANG_CODE[lang.split("_")[0]],
                    max_length=max_length,
                    max_target_length=max_target_length,
                )
                target_ids = shift_tokens_right(batch["labels"], tokenizer.pad_token_id)
                decoder_input_ids = target_ids[:, :-1].contiguous()
                labels = target_ids[:, 1:].clone()
                labels[labels == tokenizer.pad_token_id] = IGNORE_IDX
                datasets["input_ids"].append(batch["input_ids"])
                datasets["attention_mask"].append(batch["attention_mask"])
                datasets["decoder_input_ids"].append(decoder_input_ids)
                datasets["labels"].append(labels)
            dump_dataset = {
                "input_ids": torch.cat(datasets["input_ids"], 0),
                "attention_mask": torch.cat(datasets["attention_mask"], 0),
                "decoder_input_ids": torch.cat(datasets["decoder_input_ids"], 0),
                "labels": torch.cat(datasets["labels"], 0),
            }
            output_path = (
                data_dir / dataset_name / "processed" / "mbart_2mono" / f"{split}.pt"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dump_dataset, output_path.open("wb"))

            if split == "train":
                for source_lang, target_lang in [("en", "zh"), ("zh", "en")]:
                    source_file_path = (
                        data_dir
                        / dataset_name
                        / "processed"
                        / source_lang
                        / f"{split}.json"
                    )
                    source_data = json.loads(source_file_path.read_text())
                    target_file_path = (
                        data_dir
                        / dataset_name
                        / "processed"
                        / target_lang
                        / f"{split}.json"
                    )
                    target_data = json.loads(target_file_path.read_text())

                    keys = random.sample(
                        list(source_data.keys()), len(source_data) // 4
                    )
                    sources = [source_data[k]["history"] for k in keys]
                    targets = [target_data[k]["history"] for k in keys]

                    if source_lang == "zh":
                        sources = list(map(clean_chinese_spaces, sources))
                    if target_lang == "zh":
                        targets = list(map(clean_chinese_spaces, targets))

                    batch = tokenizer.prepare_seq2seq_batch(
                        src_texts=sources,
                        src_lang=LANG2LANG_CODE[source_lang],
                        tgt_texts=targets,
                        tgt_lang=LANG2LANG_CODE[target_lang],
                        max_length=max_length,
                        max_target_length=max_target_length,
                    )
                    target_ids = shift_tokens_right(
                        batch["labels"], tokenizer.pad_token_id
                    )
                    decoder_input_ids = target_ids[:, :-1].contiguous()
                    labels = target_ids[:, 1:].clone()
                    labels[labels == tokenizer.pad_token_id] = IGNORE_IDX

                    datasets["input_ids"].append(batch["input_ids"])
                    datasets["attention_mask"].append(batch["attention_mask"])
                    datasets["decoder_input_ids"].append(decoder_input_ids)
                    datasets["labels"].append(labels)

                dump_dataset = {
                    "input_ids": torch.cat(datasets["input_ids"], 0),
                    "attention_mask": torch.cat(datasets["attention_mask"], 0),
                    "decoder_input_ids": torch.cat(datasets["decoder_input_ids"], 0),
                    "labels": torch.cat(datasets["labels"], 0),
                }

            output_path = (
                data_dir
                / dataset_name
                / "processed"
                / "mbart_2mono_trans"
                / f"{split}.pt"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dump_dataset, output_path.open("wb"))


if __name__ == "__main__":
    main()

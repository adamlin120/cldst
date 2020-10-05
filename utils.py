import json
from typing import Dict, List, Union, Optional

import torch
from transformers import BertTokenizer, GPT2Tokenizer

from preprocess_multiwoz_data import SLOT_SEP, SLOT_NAME_SEP, SLOT_VALUE_SEP

IGNORE_INDEX = -100
EOS = "<eos>"
PAD = "<pad>"
BELIEF = "<BELIEF>"
ATTR_TO_SPECIAL_TOKEN = {
    "eos_token": EOS,
    "pad_token": PAD,
    "additional_special_tokens": [BELIEF, SLOT_SEP, SLOT_NAME_SEP, SLOT_VALUE_SEP],
}

SYSTEM = "system"
USER = "user"


def add_special_tokens_(model, tokenizer):
    """Add special tokens to the tokenizer and the model if they have not
    already been added."""
    orig_num_tokens = len(tokenizer)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_history_from_utterances(
    system_utterances: List[str],
    user_utterances: List[str],
) -> str:
    history = ""
    for sys, user in zip(system_utterances, user_utterances):
        if sys is not None:
            history += f"{SYSTEM} : {sys}"
        history += f"{USER} : {user}"
    return history


def stringarize_belief(
    belief: Dict[str, Dict[str, str]], add_begin_of_belief: bool
) -> str:
    belief_str = BELIEF if add_begin_of_belief else ""
    belief_str += "".join(
        [
            f"{SLOT_SEP} {domain} {SLOT_NAME_SEP} {slot_name} {SLOT_VALUE_SEP} {slot_value}"
            for domain, domain_slots in belief.items()
            for slot_name, slot_value in domain_slots.items()
        ]
    )
    return belief_str


def build_input_from_segments(
    tokenizer: Union[BertTokenizer, GPT2Tokenizer],
    system_utterances: List[str],
    user_utterances: List[str],
    belief: Optional[Dict[str, Dict[str, str]]] = None,
    add_eos: Optional[bool] = True,
) -> Dict[str, List[int]]:
    def tokenize_to_ids(x: str) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))

    history = build_history_from_utterances(system_utterances, user_utterances)

    input_ids = tokenize_to_ids(history + f" {BELIEF}")
    labels = [IGNORE_INDEX] * len(input_ids)

    if belief is not None:
        belief_str = stringarize_belief(belief, add_begin_of_belief=False)
        belief_tokens: List[int] = tokenize_to_ids(belief_str)
        input_ids += belief_tokens
        labels += belief_tokens

    if add_eos:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    assert len(input_ids) == len(labels)
    instance = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return instance


def pad_truncate_sequence(
    seq: List[List[int]], padding_value: int, max_length: int
) -> torch.LongTensor:
    max_length = min(max_length, max(len(s) for s in seq))
    padded_seq = [
        s[max(0, len(s) - max_length) :] + [padding_value] * (max_length - len(s))
        for s in seq
    ]
    padded_tensor = torch.tensor(padded_seq, dtype=torch.long)
    return padded_tensor


def load_tokenizer(
    model_checkpoint: str, add_special_token: bool
) -> Union[BertTokenizer, GPT2Tokenizer]:
    use_bert_tokenizer = "CDial" in model_checkpoint
    if use_bert_tokenizer:
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint, do_lower_case=True)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    if add_special_token:
        tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    return tokenizer


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

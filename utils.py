import json
from typing import Dict, List, Union, Optional, Tuple

import torch
from transformers import BertTokenizer, GPT2Tokenizer


IGNORE_INDEX = -100

BELIEF = "<BELIEF>"
SLOT_SEP = "<SLOT>"
SLOT_NAME_SEP = "<SLOT_NAME>"
SLOT_VALUE_SEP = "<SLOT_VALUE>"
ATTR_TO_SPECIAL_TOKEN = {
    "additional_special_tokens": [BELIEF, SLOT_SEP, SLOT_NAME_SEP, SLOT_VALUE_SEP],
}


LANG_CODE = {"en": "en_XX", "zh": "zh_CN"}


def build_history_from_utterances(
    system_utterances: List[str], user_utterances: List[str], lang: str
) -> str:
    system, user = {"en": ("system", "user"), "zh": ("用戶", "系統")}[lang]
    history = ""
    for sys, usr in zip(system_utterances, user_utterances):
        if sys is not None:
            history += f"{system} : {sys} "
        history += f"{user} : {usr} "
    return history.strip()


def stringarize_belief(
    belief: Dict[str, Dict[str, str]], add_begin_of_belief: bool
) -> str:
    belief_str = BELIEF if add_begin_of_belief else ""
    belief_str += "".join(
        [
            f"{SLOT_SEP} {domain} {SLOT_NAME_SEP} {slot_name} {SLOT_VALUE_SEP} {slot_value}"
            for domain, domain_slots in belief.items()
            for slot_name, slot_value in domain_slots.items()
            if slot_value.strip()
        ]
    )
    return belief_str


def build_lm_sequence(
    tokenizer: Union[BertTokenizer, GPT2Tokenizer],
    system_utterances: List[str],
    user_utterances: List[str],
    lang: str,
    belief: Optional[Dict[str, Dict[str, str]]] = None,
    add_eos: Optional[bool] = True,
) -> Dict[str, List[int]]:
    def tokenize_to_ids(x: str) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))

    history = build_history_from_utterances(system_utterances, user_utterances, lang)

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


def pad_back_or_truncate_start_sequence(
    seq: List[List[int]], padding_value: int, max_length: int
) -> torch.LongTensor:
    max_length = min(max_length, max(len(s) for s in seq))
    padded_seq = [
        s[max(0, len(s) - max_length) :] + [padding_value] * (max_length - len(s))
        for s in seq
    ]
    padded_tensor = torch.tensor(padded_seq, dtype=torch.long)
    return padded_tensor


def get_history_utterances(
    turns: List[Tuple[str, str, Dict[str, Dict[str, str]]]],
    num_history_turns: int,
) -> Tuple[List[str], List[str]]:
    if isinstance(num_history_turns, int):
        system_utterances, user_utterances, _ = zip(*turns)
        if num_history_turns > 0:
            system_utterances = system_utterances[-num_history_turns:]
            user_utterances = user_utterances[-num_history_turns:]
    else:
        raise ValueError(
            f"num_history_turns: {num_history_turns} should be -1 or positive integer"
        )
    return system_utterances, user_utterances


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


def add_special_tokens_to_model_tokenizer(model, tokenizer) -> None:
    """Add special tokens to the tokenizer and the model if they have not
    already been added."""
    orig_num_tokens = len(tokenizer)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

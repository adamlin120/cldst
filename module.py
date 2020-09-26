from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import LightningModule
from transformers import AdamW, BertTokenizer, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_INDEX = -100
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
BELIEF = "<BELIEF>"
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": BOS,
    "eos_token": EOS,
    "pad_token": PAD,
    "additional_special_tokens": [BELIEF],
}


class ConditionalLM(LightningModule):
    def __init__(
        self,
        hparams: Namespace,
    ):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = BertTokenizer.from_pretrained(
            self.hparams.model_checkpoint, do_lower_case=True
        )
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_checkpoint)
        # Add special tokens if they are not already added
        add_special_tokens_(self.model, self.tokenizer)

    def forward(self, *args, **kwargs) -> CausalLMOutputWithPast:
        return self.model.forward(return_dict=True, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        tensorboard_logs = {"train_loss": output.loss}
        return {"loss": output.loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        return {"val_loss": output.loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_ppl": avg_loss.exp(),
        }
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        output = self.validation_epoch_end(*args, **kwargs)
        output = {k.replace("val", "test"): v for k, v in output.items()}
        output["log"] = {k.replace("val", "test"): v for k, v in output["log"].items()}
        return output

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.hparams.lr, correct_bias=True
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="models/CDial-GPT2_LCCC-base/",
            help="Dir path to pretrained model",
        )
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        return parser


def add_special_tokens_(model, tokenizer):
    """Add special tokens to the tokenizer and the model if they have not
    already been added."""
    orig_num_tokens = len(tokenizer)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

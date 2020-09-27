from argparse import ArgumentParser, Namespace

from pytorch_lightning import LightningModule, TrainResult, EvalResult
from transformers import AdamW, BertTokenizer, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_INDEX = -100
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
BELIEF = "<BELIEF>"
SLOT_SEP = "<SLOT>"
SLOT_NAME_SEP = "<SLOT_NAME>"
SLOT_VALUE_SEP = "<SLOT_VALUE>"
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": BOS,
    "eos_token": EOS,
    "pad_token": PAD,
    "additional_special_tokens": [BELIEF, SLOT_SEP, SLOT_NAME_SEP, SLOT_VALUE_SEP],
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

    def training_step(self, batch, batch_idx) -> TrainResult:
        output = self.forward(**batch)
        result = TrainResult(
            minimize=output.loss,
        )
        result.log("train_loss", output.loss.detach(), prog_bar=True, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx) -> EvalResult:
        output = self.forward(**batch)
        result = EvalResult(checkpoint_on=output.loss, early_stop_on=output.loss)
        result.log(
            "val_loss",
            output.loss,
            prog_bar=True,
        )
        return result

    def test_step(self, batch, batch_idx) -> EvalResult:
        output = self.forward(**batch)
        result = EvalResult(checkpoint_on=output.loss, early_stop_on=output.loss)
        result.log(
            "test_loss",
            output.loss,
            prog_bar=True,
        )
        return result

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

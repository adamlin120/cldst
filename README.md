# Cross Lingual DST

## Data
```bash
bash download_data.sh
```

## Model
Two approachs for generative dst: autoregressive language model and seq2seq.
Please refer `conditional_lm.py` for the former and `mbart.py` for the latter.

## Experiment Log
[Wandb logs](https://wandb.ai/ytlin/xldst?workspace=user-ytlin) shows stats for training and evaluation.

CLI commands are also recorded in ecah run page. (e.g. [run](https://wandb.ai/ytlin/xldst/runs/3gkn5z6t/overview?workspace=user-ytlin))

## Requirements
```
transformers==3.2.0
pytorch-lightning==0.9.0
wandb
```

import json
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from langchain.text_splitter import SpacyTextSplitter
from omegaconf import OmegaConf

from transformers import DataCollatorForLanguageModeling
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def text_splitter(document: str, max_length: int = 512) -> list[str]:
    text_splitter = SpacyTextSplitter(separator="[SEP]")
    docs = text_splitter.split_text(document.replace("\n", ""))

    chunks = []
    chunk = ""
    for text in docs[0].split("[SEP]"):
        if len(chunk) + len(text) > max_length:
            chunks.append(chunk)
            chunk = text
        else:
            chunk += text
    if chunk:
        chunks.append(chunk)
    return chunks


# データセット読み込み
with open("arxiv.jsonl", "r") as f:
    texts = [json.loads(line)["text"] for line in f][:5]
    # dataset = Dataset.from_list([json.loads(line) for line in f][:5])

dataset_texts = []
for text in texts:
    dataset_texts.extend(text_splitter(text))
dataset = Dataset.from_list([{"text": text} for text in dataset_texts])

# コンフィグ読み込み
config_path = "./train_base.yaml"
config = OmegaConf.load(config_path)

# モデルの定義
model = AutoModelForCausalLM.from_pretrained(
    config.model.model, torch_dtype=torch.float16, use_cache=config.model.use_cache
)
tokenizer = AutoTokenizer.from_pretrained(
    config.model.tokenizer,
    add_eos_token=True,  # EOSの追加を指示 defaultはFalse
)

dataset = dataset.map(lambda data: tokenizer(data["text"]))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 学習
tokenizer.pad_token = tokenizer.eos_token
training_args = TrainingArguments(**config.train)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

with torch.autocast("cuda"):
    trainer.train()

trainer.save_model("output")

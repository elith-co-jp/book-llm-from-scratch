import json
import os
import warnings

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# データセット読み込み
with open("chunked_dataset.jsonl", "r") as f:
    dataset = Dataset.from_list([json.loads(line) for line in f])


# トークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# モデルの定義
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=512,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# データセットのトークン化
dataset = dataset.map(
    lambda data: tokenizer(data["text"], truncation=True, max_length=512), batched=True
)


# 学習
training_args = TrainingArguments(
    output_dir="./output",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    num_train_epochs=1,
    per_device_train_batch_size=3,
    learning_rate=1e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    optim="adamW_torch",
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

with torch.autocast("cuda"):
    trainer.train()

# モデルの保存
trainer.save_model("pretrained_model")

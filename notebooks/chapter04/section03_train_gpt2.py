"""
4.3 分散学習による事前学習: rinnaの事前学習済みGPT-2モデルで継続学習

このスクリプトは、4.1節の前処理済みデータを用いて、
rinnaの事前学習済み日本語GPT-2モデル (`rinna/japanese-gpt2-medium`) を
Causal Language Modeling でファインチューニングします。

実行方法:
---------
単一GPUでの実行:
    python section03_train_gpt2.py

複数GPUでのデータ並列実行 (4 GPU):
    torchrun --nproc_per_node=4 section03_train_gpt2.py

特定のGPUを指定する場合:
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 section03_train_gpt2.py

DeepSpeed (ZeRO Stage 2) での実行:
    deepspeed --num_gpus=4 section03_train_gpt2.py --deepspeed ds_config.json

必要なデータ:
    - data/aozora_preprocessed (4.1節で作成した前処理済みデータ)
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 設定
# =============================================================================

# モデル設定
MODEL_NAME = "rinna/japanese-gpt2-medium"

# 学習パラメータ
BLOCK_SIZE = 512
EVAL_RATIO = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
NUM_TRAIN_EPOCHS = 3

# データ設定
TEXT_COL = "text"


class TrainingProgressCallback(TrainerCallback):
    """学習の進捗状況をログ出力するコールバック"""

    def __init__(self, local_rank, total_steps):
        self.local_rank = local_rank
        self.total_steps = total_steps
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.local_rank == 0:
            import time
            self.start_time = time.time()
            logger.info("\n" + "=" * 70)
            logger.info("学習開始")
            logger.info(f"総ステップ数: {self.total_steps}")
            logger.info(f"エポック数: {args.num_train_epochs}")
            logger.info(f"GPU数: {args.world_size}")
            logger.info(f"バッチサイズ/GPU: {args.per_device_train_batch_size}")
            logger.info(f"勾配蓄積: {args.gradient_accumulation_steps}")
            effective_batch = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
            logger.info(f"実効バッチサイズ: {effective_batch}")
            logger.info("=" * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.local_rank == 0 and logs is not None:
            import time
            elapsed = time.time() - self.start_time if self.start_time else 0
            elapsed_str = f"{int(elapsed // 60)}分{int(elapsed % 60)}秒"

            step = state.global_step
            progress = (step / self.total_steps) * 100 if self.total_steps > 0 else 0

            # 残り時間の推定
            if step > 0 and elapsed > 0:
                time_per_step = elapsed / step
                remaining_steps = self.total_steps - step
                eta_seconds = time_per_step * remaining_steps
                eta_str = f"{int(eta_seconds // 60)}分{int(eta_seconds % 60)}秒"
            else:
                eta_str = "計算中..."

            log_msg = f"\n[Step {step}/{self.total_steps} ({progress:.1f}%)] "
            log_msg += f"経過: {elapsed_str} | 残り: {eta_str}"

            if "loss" in logs:
                log_msg += f"\n  train_loss: {logs['loss']:.4f}"
            if "eval_loss" in logs:
                log_msg += f" | eval_loss: {logs['eval_loss']:.4f}"
            if "learning_rate" in logs:
                log_msg += f" | lr: {logs['learning_rate']:.2e}"

            logger.info(log_msg)

    def on_train_end(self, args, state, control, **kwargs):
        if self.local_rank == 0:
            import time
            total_time = time.time() - self.start_time if self.start_time else 0
            logger.info("\n" + "=" * 70)
            logger.info("学習完了")
            logger.info(f"総学習時間: {int(total_time // 60)}分{int(total_time % 60)}秒")
            logger.info(f"最終ステップ: {state.global_step}")
            logger.info("=" * 70)


class GenerationCallback(TrainerCallback):
    """学習中に定期的にテキスト生成を行うコールバック"""

    def __init__(self, tokenizer, test_prompts, generation_interval, local_rank):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_interval = generation_interval
        self.local_rank = local_rank
        self.has_generated_initial = False

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.has_generated_initial and self.local_rank == 0:
            self._generate_samples(model, 0)
            self.has_generated_initial = True

    def on_log(self, args, state, control, model=None, **kwargs):
        current_step = state.global_step
        if (
            current_step > 0
            and current_step % self.generation_interval == 0
            and self.local_rank == 0
        ):
            self._generate_samples(model, current_step)

    def _generate_samples(self, model, step):
        """テストプロンプトで生成サンプルを表示"""
        logger.info("\n" + "=" * 70)
        logger.info(f"【生成サンプル】 Step {step}")
        logger.info("=" * 70)

        was_training = model.training
        model.eval()

        # モデルが配置されているデバイスを取得
        device = next(model.parameters()).device

        for i, prompt in enumerate(self.test_prompts, 1):
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # プロンプト部分と生成部分を分けて表示
            generated_part = generated_text[len(prompt):]
            logger.info(f"\n[{i}] プロンプト: {prompt}")
            logger.info(f"    生成結果: {generated_part[:100]}...")

        logger.info("\n" + "=" * 70 + "\n")

        if was_training:
            model.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT-2モデルの分散学習スクリプト"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/aozora_preprocessed",
        help="前処理済みデータのディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/rinna-gpt2-aozora-finetuned",
        help="モデルの保存先ディレクトリ",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed設定ファイルのパス (例: ds_config.json)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="BF16混合精度学習を有効化",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分散学習時のローカルランク (torchrunが自動設定)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ローカルランクの取得 (torchrun/deepspeed が設定)
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0

    # メインプロセスのみログ出力
    is_main_process = local_rank in [-1, 0]

    if is_main_process:
        logger.info("=" * 60)
        logger.info("4.3 分散学習による事前学習")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # データセットの読み込み (4.3.3 データセットの準備)
    # -------------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"前処理済みデータが見つかりません: {data_dir}\n"
            "section01_dataset_preprocessing.ipynb を先に実行してください。"
        )

    dataset = load_from_disk(str(data_dir))
    if is_main_process:
        logger.info(f"読み込んだデータセット: {dataset}")
        logger.info(f"サンプル数: {len(dataset)}")

    # train/eval に分割
    split_ds = dataset.train_test_split(test_size=EVAL_RATIO, seed=42)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]

    if is_main_process:
        logger.info(f"訓練データ: {len(train_dataset)} サンプル")
        logger.info(f"評価データ: {len(eval_dataset)} サンプル")

    # -------------------------------------------------------------------------
    # トークナイザの読み込み (4.3.2.1 事前学習済みトークナイザの利用)
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        logger.info(f"語彙サイズ: {len(tokenizer)}")
        logger.info(f"BOS token: {tokenizer.bos_token}")
        logger.info(f"EOS token: {tokenizer.eos_token}")
        logger.info(f"PAD token: {tokenizer.pad_token}")

    # -------------------------------------------------------------------------
    # データセットのトークン化
    # -------------------------------------------------------------------------
    def tokenize_function(examples):
        return tokenizer(
            examples[TEXT_COL],
            truncation=True,
            max_length=BLOCK_SIZE,
        )

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=[TEXT_COL]
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=[TEXT_COL]
    )

    # -------------------------------------------------------------------------
    # モデルの読み込み (4.3.2.2 事前学習済みモデルの利用)
    # -------------------------------------------------------------------------
    config = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)
    model.config.pad_token_id = tokenizer.pad_token_id

    if is_main_process:
        logger.info(f"モデル: {MODEL_NAME}")
        logger.info(f"語彙サイズ: {config.vocab_size}")
        logger.info(f"最大シーケンス長: {config.n_positions}")
        logger.info(f"レイヤー数: {config.n_layer}")
        logger.info(f"隠れ層次元: {config.n_embd}")

    # -------------------------------------------------------------------------
    # 学習設定 (4.3.4 Trainer を用いたデータ並列学習)
    # -------------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ログ・評価・保存間隔の計算
    train_dataset_size = len(tokenized_train)
    steps_per_epoch = train_dataset_size // (
        PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    )
    logging_steps = max(1, steps_per_epoch // 10)

    if is_main_process:
        logger.info(f"\nデータセットサイズ: {train_dataset_size}")
        logger.info(f"1エポックあたりのステップ数: {steps_per_epoch}")
        logger.info(f"ログ・評価・保存間隔: {logging_steps} steps")

    # TrainingArguments (4.3.4.1 学習設定)
    # グローバルバッチサイズ = per_device_train_batch_size × GPU数 × gradient_accumulation_steps
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=logging_steps,
        save_steps=logging_steps,
        save_total_limit=3,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,  # DDP最適化
        report_to=["none"],
        load_best_model_at_end=False,
        # DeepSpeed / 混合精度設定 (4.3.5 DeepSpeed による ZeRO の活用)
        deepspeed=args.deepspeed,
        bf16=args.bf16,
    )

    # テスト用プロンプト
    test_prompts = [
        "吾輩は猫である。名前はまだ無い。",
        "明治時代の",
        "東京の街には",
        "先生は言った。「",
    ]

    # 総ステップ数の計算
    total_steps = (
        len(tokenized_train)
        // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        * NUM_TRAIN_EPOCHS
    )

    # コールバックの作成
    progress_callback = TrainingProgressCallback(
        local_rank=local_rank,
        total_steps=total_steps,
    )
    generation_callback = GenerationCallback(
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        generation_interval=logging_steps,
        local_rank=local_rank,
    )

    # -------------------------------------------------------------------------
    # 学習実行 (4.3.4.2 学習の実行)
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[progress_callback, generation_callback],
    )

    if is_main_process:
        if args.deepspeed:
            logger.info(f"DeepSpeed設定: {args.deepspeed}")
        else:
            logger.info("データ並列 (DDP) モードで実行")

    trainer.train()

    # モデルの保存
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_main_process:
        logger.info(f"\nモデルを保存しました: {args.output_dir}")
        logger.info("=" * 60)
        logger.info("学習完了")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

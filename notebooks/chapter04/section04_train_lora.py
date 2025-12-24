"""
4.4 学習の効率化（LoRA）: HuggingFace PEFTを用いたLoRAファインチューニング

このスクリプトは、4.1節の前処理済みデータを用いて、
rinnaの事前学習済み日本語GPT-2モデル (`rinna/japanese-gpt2-medium`) を
LoRA (Low-Rank Adaptation) でファインチューニングします。

実行方法:
---------
単一GPUでの実行:
    python section04_train_lora.py

特定のGPUを指定する場合:
    CUDA_VISIBLE_DEVICES=0 python section04_train_lora.py

LoRAパラメータをカスタマイズする場合:
    python section04_train_lora.py --lora_rank 16 --lora_alpha 32

W&Bでメトリクスを記録する場合:
    python section04_train_lora.py --wandb_project gpt2-lora-aozora

必要なデータ:
    - data/aozora_preprocessed (4.1節で作成した前処理済みデータ)

依存ライブラリ:
    pip install peft
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
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
LEARNING_RATE = 1e-4  # LoRAは通常より高い学習率を使用
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
NUM_TRAIN_EPOCHS = 3

# LoRAデフォルト設定
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

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
        # eval_loss がある時のみログ出力（評価後に1回だけ）
        if self.local_rank == 0 and logs is not None and "eval_loss" in logs:
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
            log_msg += f" | eval_loss: {logs['eval_loss']:.4f}"

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


class BestModelCallback(TrainerCallback):
    """eval_lossが改善されたらLoRAアダプターを上書き保存するコールバック"""

    def __init__(self, output_dir, tokenizer, local_rank):
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.best_eval_loss = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if self.local_rank == 0 and metrics is not None:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None and eval_loss < self.best_eval_loss:
                logger.info(f"\n★ ベストモデル更新: eval_loss {self.best_eval_loss:.4f} → {eval_loss:.4f}")
                self.best_eval_loss = eval_loss
                # LoRAアダプターのみ保存
                model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                logger.info(f"  保存先: {self.output_dir}")


class WandbMetricsCallback(TrainerCallback):
    """W&BにGPUメモリ使用量などの追加メトリクスを記録するコールバック"""

    def __init__(self, local_rank):
        self.local_rank = local_rank

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.local_rank == 0 and logs is not None:
            # GPUメモリ使用量を記録
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                gpu_max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

                wandb.log({
                    "gpu/memory_allocated_gb": gpu_memory_allocated,
                    "gpu/memory_reserved_gb": gpu_memory_reserved,
                    "gpu/max_memory_allocated_gb": gpu_max_memory,
                }, commit=False)

            # perplexityを計算して記録
            if "loss" in logs:
                wandb.log({"train/perplexity": 2.71828 ** logs["loss"]}, commit=False)
            if "eval_loss" in logs:
                wandb.log({"eval/perplexity": 2.71828 ** logs["eval_loss"]}, commit=False)


class GenerationCallback(TrainerCallback):
    """学習中に定期的にテキスト生成を行うコールバック"""

    def __init__(self, tokenizer, test_prompts, generation_interval, local_rank, output_dir):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_interval = generation_interval
        self.local_rank = local_rank
        self.has_generated_initial = False
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "generation_samples.csv"

        # CSVファイルの初期化（ヘッダー書き込み）
        if self.local_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("step,prompt,generated_text,timestamp\n")

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.has_generated_initial and self.local_rank == 0:
            self._generate_samples(model, 0)
            self.has_generated_initial = True

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        current_step = state.global_step
        # eval_loss がある時のみ生成（評価後に1回だけ実行）
        if (
            current_step > 0
            and current_step % self.generation_interval == 0
            and self.local_rank == 0
            and logs is not None
            and "eval_loss" in logs
        ):
            self._generate_samples(model, current_step)

    def _generate_samples(self, model, step):
        """テストプロンプトで生成サンプルを表示し、CSVに記録"""
        import csv
        from datetime import datetime

        logger.info("\n" + "=" * 70)
        logger.info(f"【生成サンプル】 Step {step}")
        logger.info("=" * 70)

        was_training = model.training
        model.eval()

        # モデルが配置されているデバイスを取得
        device = next(model.parameters()).device
        timestamp = datetime.now().isoformat()

        # CSVに追記
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

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

                # CSVに記録
                writer.writerow([step, prompt, generated_part[:200], timestamp])

        logger.info("\n" + "=" * 70 + "\n")

        if was_training:
            model.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT-2モデルのLoRAファインチューニングスクリプト"
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
        default="./models/rinna-gpt2-aozora-lora",
        help="モデルの保存先ディレクトリ",
    )
    # LoRA設定
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=LORA_RANK,
        help=f"LoRAのランク（低ランク行列の次元数）（デフォルト: {LORA_RANK}）",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRAのスケーリングファクター（デフォルト: {LORA_ALPHA}）",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=LORA_DROPOUT,
        help=f"LoRA層のドロップアウト率（デフォルト: {LORA_DROPOUT}）",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=LORA_TARGET_MODULES,
        help=f"LoRAを適用する層（デフォルト: {LORA_TARGET_MODULES}）",
    )
    # 学習設定
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"学習率（デフォルト: {LEARNING_RATE}）",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_TRAIN_EPOCHS,
        help=f"エポック数（デフォルト: {NUM_TRAIN_EPOCHS}）",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="BF16混合精度学習を有効化",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="FP16混合精度学習を有効化",
    )
    # W&B設定
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&Bプロジェクト名 (指定するとW&Bログが有効化)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&Bのrun名",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="ログ記録のステップ間隔 (デフォルト: 1エポックの1/10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from datetime import datetime

    # ローカルランクの取得（単一GPU想定だが、将来の拡張に備える）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # タイムスタンプ付きの出力ディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"出力ディレクトリ: {output_dir}")

    # W&B の初期化（メインプロセスのみ）
    use_wandb = args.wandb_project is not None and is_main_process
    if use_wandb:
        run_name = args.wandb_run_name or f"gpt2-lora-r{args.lora_rank}-a{args.lora_alpha}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": MODEL_NAME,
                "block_size": BLOCK_SIZE,
                "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": args.learning_rate,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
                "num_train_epochs": args.num_epochs,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,
                "bf16": args.bf16,
                "fp16": args.fp16,
            },
        )
        logger.info(f"W&B プロジェクト: {args.wandb_project}")
        logger.info(f"W&B run名: {run_name}")

    if is_main_process:
        logger.info("=" * 60)
        logger.info("4.4 学習の効率化（LoRA）")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # データセットの読み込み
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
    # トークナイザの読み込み
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
    # モデルの読み込みとLoRA適用 (4.4.5 HuggingFace PEFTを用いたLoRA学習)
    # -------------------------------------------------------------------------
    config = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)
    model.config.pad_token_id = tokenizer.pad_token_id

    if is_main_process:
        logger.info(f"\nベースモデル: {MODEL_NAME}")
        logger.info(f"語彙サイズ: {config.vocab_size}")
        logger.info(f"最大シーケンス長: {config.n_positions}")
        logger.info(f"レイヤー数: {config.n_layer}")
        logger.info(f"隠れ層次元: {config.n_embd}")

    # LoRA設定
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )

    # LoRAをモデルに適用
    model = get_peft_model(model, lora_config)

    if is_main_process:
        logger.info("\n" + "-" * 60)
        logger.info("LoRA設定")
        logger.info("-" * 60)
        logger.info(f"ランク (r): {args.lora_rank}")
        logger.info(f"スケーリング (alpha): {args.lora_alpha}")
        logger.info(f"ドロップアウト: {args.lora_dropout}")
        logger.info(f"適用層: {args.lora_target_modules}")
        logger.info("-" * 60)
        model.print_trainable_parameters()
        logger.info("-" * 60)

    # -------------------------------------------------------------------------
    # 学習設定
    # -------------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ログ・評価・保存間隔の計算
    train_dataset_size = len(tokenized_train)
    steps_per_epoch = train_dataset_size // (
        PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    )
    # 引数で指定されていれば使用、なければ1エポックの1/10
    logging_steps = args.logging_steps if args.logging_steps else max(1, steps_per_epoch // 10)

    if is_main_process:
        logger.info(f"\nデータセットサイズ: {train_dataset_size}")
        logger.info(f"1エポックあたりのステップ数: {steps_per_epoch}")
        logger.info(f"ログ・評価・保存間隔: {logging_steps} steps")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=logging_steps,
        save_strategy="no",  # チェックポイントは保存しない（BestModelCallbackで保存）
        dataloader_num_workers=4,
        report_to=["wandb"] if use_wandb else ["none"],
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        bf16=args.bf16,
        fp16=args.fp16,
        run_name=args.wandb_run_name if use_wandb else None,
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
        * args.num_epochs
    )

    # コールバックの作成
    callbacks = []

    progress_callback = TrainingProgressCallback(
        local_rank=local_rank,
        total_steps=total_steps,
    )
    callbacks.append(progress_callback)

    generation_callback = GenerationCallback(
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        generation_interval=logging_steps,
        local_rank=local_rank,
        output_dir=output_dir,
    )
    callbacks.append(generation_callback)

    # ベストモデル保存コールバック
    best_model_callback = BestModelCallback(
        output_dir=output_dir,
        tokenizer=tokenizer,
        local_rank=local_rank,
    )
    callbacks.append(best_model_callback)

    # W&Bが有効な場合、追加メトリクスコールバックを追加
    if use_wandb:
        wandb_metrics_callback = WandbMetricsCallback(local_rank=local_rank)
        callbacks.append(wandb_metrics_callback)

    # -------------------------------------------------------------------------
    # 学習実行
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if is_main_process:
        logger.info("\n学習モードで実行（LoRA）")

    trainer.train()

    # LoRAアダプターはBestModelCallbackで保存済み
    if is_main_process:
        logger.info(f"\nベストLoRAアダプター保存先: {output_dir}")
        logger.info("=" * 60)
        logger.info("学習完了")
        logger.info("=" * 60)

        # W&Bの終了処理
        if use_wandb:
            # 最終的なサマリーを記録
            train_result = trainer.state.log_history
            if train_result:
                # 最終の学習結果を取得
                final_metrics = {}
                for log in reversed(train_result):
                    if "train_runtime" in log:
                        final_metrics["total_train_runtime_sec"] = log["train_runtime"]
                        final_metrics["train_samples_per_second"] = log["train_samples_per_second"]
                        final_metrics["train_steps_per_second"] = log["train_steps_per_second"]
                        break

                for log in reversed(train_result):
                    if "eval_loss" in log:
                        final_metrics["final_eval_loss"] = log["eval_loss"]
                        final_metrics["final_perplexity"] = 2.71828 ** log["eval_loss"]
                        break

                if final_metrics:
                    wandb.log(final_metrics)
                    wandb.summary.update(final_metrics)

            wandb.finish()
            logger.info("W&B ログを終了しました")


if __name__ == "__main__":
    main()

"""夏目漱石のテキストでGPTを訓練（nanoGPTと同様）。"""

import os
import re
import torch
import requests
from bs4 import BeautifulSoup
from llm_from_scratch.gpt import (
    GPT, GPTConfig, SimpleTokenizer,
    create_dataloaders, GPTTrainer
)


def download_soseki():
    """夏目漱石のテキストをダウンロード。"""
    url = "https://www.aozora.gr.jp/cards/000148/files/789_14547.html"

    # データディレクトリを作成
    os.makedirs("data/soseki", exist_ok=True)

    # ダウンロードして処理
    filepath = "data/soseki/input.txt"
    if not os.path.exists(filepath):
        print("夏目漱石のテキストをダウンロード中...")
        response = requests.get(url)
        response.encoding = 'shift_jis'  # 青空文庫はShift_JIS

        # BeautifulSoupでHTMLを解析
        soup = BeautifulSoup(response.text, 'html.parser')

        # 本文を抽出（青空文庫の構造に基づく）
        main_text = soup.find('div', class_='main_text')
        if main_text:
            text = main_text.get_text()
        else:
            # フォールバック：bodyからテキストを抽出
            text = soup.get_text()

        # 青空文庫の注記や記号を除去
        text = re.sub(r'［＃.*?］', '', text)  # 注記を除去
        text = re.sub(r'《.*?》', '', text)    # ルビを除去
        text = re.sub(r'｜', '', text)        # 縦線を除去
        text = re.sub(r'　', ' ', text)       # 全角スペースを半角に
        text = re.sub(r'\n+', '\n', text)     # 連続する改行を一つに
        text = text.strip()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"ダウンロード完了: {filepath}")

    # テキストを読み込み
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"データセットサイズ: {len(text)} 文字")
    return text


def train_soseki_gpt():
    """夏目漱石のテキストでGPTを訓練（文字レベル）。"""

    # データセットをダウンロードして読み込み
    text = download_soseki()

    # 文字レベルのトークナイザーを作成
    tokenizer = SimpleTokenizer(text)
    print(f"語彙数: {tokenizer.vocab_size} ユニーク文字")

    # データローダーを作成
    train_loader, val_loader = create_dataloaders(
        text, tokenizer,
        block_size=256,      # コンテキスト長
        batch_size=64,       # バッチサイズ
        train_split=0.9
    )

    # モデル設定（nanoGPTのshakespeareと同様）
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=384,          # 埋め込み次元
        n_layer=6,           # レイヤー数
        n_head=6,            # アテンションヘッド数
        block_size=256,      # コンテキストウィンドウ
        dropout=0.2
    )

    print(f"モデルサイズ: 約{config.get_model_size():.2f}Mパラメータ")

    # モデルを作成
    model = GPT(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout
    )

    # トレーナーを作成
    trainer = GPTTrainer(
        model, train_loader, val_loader,
        learning_rate=1e-3,
        weight_decay=0.1,
        warmup_steps=100,
        max_steps=5000,      # 5000ステップ訓練
        grad_clip=1.0
    )

    # モデルを訓練
    print("\n訓練開始...")
    losses = trainer.train(log_interval=100, eval_interval=500)

    # サンプルテキストを生成
    print("\n" + "="*50)
    print("夏目漱石風のテキストを生成中...")
    print("="*50)

    model.eval()

    # プロンプトで開始
    prompts = [
        "吾輩は猫である",
        "明治",
        "東京",
        "先生"
    ]

    for prompt in prompts:
        print(f"\nプロンプト: {prompt}")
        print("-" * 30)

        # プロンプトをエンコード
        prompt_tokens = torch.tensor(
            tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(trainer.device)

        # テキストを生成
        with torch.no_grad():
            generated = model.generate(
                prompt_tokens,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40
            )

        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        print(generated_text)

    # モデルを保存
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/soseki_gpt_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nモデル保存: {checkpoint_path}")

    return model, tokenizer, losses


if __name__ == "__main__":
    # 夏目漱石で訓練
    model, tokenizer, losses = train_soseki_gpt()
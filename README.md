# LLM from Scratch

大規模言語モデル（LLM）の基礎から実装までを学ぶためのリポジトリです。TransformerアーキテクチャからGPTモデルの実装まで、段階的に理解を深めることができます。

## 📚 ドキュメント構成

### Chapter 2: Transformerアーキテクチャ
Transformerの基本的な仕組みと実装について学びます。

### Chapter 3: GPTモデル
- `3_1_GPTモデルの概要.md` - GPTの基本概念と進化の歴史
- `3_2_Tokenizerと入力処理.md` - テキストの前処理とトークン化
- `3_3_GPTモデルの学習.md` - GPTモデルの学習プロセスと実装
- `3_4_他のLLMの紹介.md` - その他の大規模言語モデル

## 🚀 クイックスタート

### 環境セットアップ（uv使用）

```bash
# リポジトリのクローン
git clone https://github.com/elith-co-jp/book-llm-from-scratch.git
cd book-llm-from-scratch

# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストールと仮想環境の作成
uv sync

# 仮想環境の有効化
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### GPTモデルの学習例

```python
from llm_from_scratch.gpt import (
    GPT, GPTConfig, SimpleTokenizer,
    create_dataloaders, GPTTrainer
)

# テキストデータの準備
text = "Your training text here..."

# トークナイザーの作成
tokenizer = SimpleTokenizer(text)

# データローダーの作成
train_loader, val_loader = create_dataloaders(
    text, tokenizer,
    block_size=64,
    batch_size=8
)

# モデルの設定と作成
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=128,
    n_layer=4,
    n_head=4,
    block_size=64
)

model = GPT(
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    n_layer=config.n_layer,
    n_head=config.n_head,
    block_size=config.block_size
)

# 学習の実行
trainer = GPTTrainer(model, train_loader, val_loader)
trainer.train()
```

詳細な実行例は `examples/train_gpt.py` を参照してください。

```bash
# サンプルスクリプトの実行
uv run python examples/train_gpt.py
```

## 📂 プロジェクト構造

```
book-llm-from-scratch/
├── docs/                    # ドキュメント
│   ├── chapter02/          # Transformerの解説
│   └── chapter03/          # GPTモデルの解説
├── llm_from_scratch/       # 実装コード
│   ├── transformer/        # Transformerの実装
│   └── gpt/               # GPTモデルの実装
│       ├── model.py       # GPTアーキテクチャ
│       ├── tokenizer.py   # トークナイザー
│       ├── dataset.py     # データセットとローダー
│       └── trainer.py     # 学習ユーティリティ
├── examples/              # 実行例
│   └── train_gpt.py      # GPT学習のサンプル
├── notebooks/             # Jupyter notebooks
├── tests/                # テストコード
└── pyproject.toml        # プロジェクト設定と依存関係
```

## 🔧 主要な機能

### GPTモデル実装
- **マルチヘッドアテンション**: 効率的な文脈理解
- **Transformerブロック**: レイヤー正規化とresidual connection
- **位置エンコーディング**: シーケンス内の位置情報
- **因果的マスク**: 自己回帰的な生成のための未来情報のマスキング

### 学習機能
- **AdamWオプティマイザー**: 適応的学習率と重み減衰
- **学習率スケジューリング**: ウォームアップとコサイン減衰
- **勾配クリッピング**: 学習の安定化
- **チェックポイント**: モデルの保存と復元

### テキスト生成
- **Temperature sampling**: 生成の多様性制御
- **Top-k sampling**: 高確率トークンからのサンプリング
- **自己回帰生成**: 文脈に基づく逐次的なトークン生成

## 📊 モデル設定例

### 小規模モデル（学習・実験用）
```python
config = GPTConfig(
    vocab_size=1000,
    n_embd=128,
    n_layer=4,
    n_head=4,
    block_size=64
)
# ~0.5M parameters
```

### 中規模モデル
```python
config = GPTConfig(
    vocab_size=50257,
    n_embd=768,
    n_layer=12,
    n_head=12,
    block_size=1024
)
# ~124M parameters (GPT-2 small相当)
```

## 🛠️ 開発環境

### 必要な依存関係

このプロジェクトは[uv](https://github.com/astral-sh/uv)を使用して依存関係を管理しています。主な依存関係：

- Python >= 3.9, < 3.12
- PyTorch >= 2.1.1
- NumPy >= 1.26.2
- matplotlib >= 3.8.2
- tqdm >= 4.66.2
- TensorBoard >= 2.12.0
- Jupyter Notebook >= 7.0.6

### uvを使った開発環境のセットアップ

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync

# 開発用依存関係も含めてインストール
uv sync --dev

# Jupyterノートブックの起動
uv run jupyter notebook

# コードのフォーマット（開発時）
uv run black llm_from_scratch/
uv run isort llm_from_scratch/

# テストの実行（開発時）
uv run pytest tests/
```

### Python環境の管理

```bash
# 新しいパッケージの追加
uv add package_name

# 開発用パッケージの追加
uv add --dev package_name

# 特定のPythonバージョンを使用
uv python pin 3.11
```

## 🔗 参考資料

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathyによる最小限のGPT実装
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformerの原論文
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) - GPT-2の論文
- [uv](https://github.com/astral-sh/uv) - 高速なPythonパッケージマネージャー

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 貢献

Issue報告やPull Requestを歓迎します。大きな変更を行う場合は、まずIssueを開いて変更内容について議論してください。

## ✉️ お問い合わせ

質問や提案がある場合は、GitHubのIssueページからお問い合わせください。
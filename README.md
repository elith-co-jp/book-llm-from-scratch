# LLM from Scratch

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

本リポジトリは書籍『[**作ってわかる大規模言語モデルの仕組み**](https://www.amazon.co.jp/dp/4296205250/)』（日経BP、2026年）の公式サポートリポジトリです。

大規模言語モデル（LLM）の基礎から実装までを、Transformerアーキテクチャ → GPTモデル → 事前学習 → アラインメントと段階的に学ぶことができます。

> 本書の正誤訂正情報は[こちら](refs/errors.md)をご覧ください。

---

## 目次

- [リポジトリ構成](#リポジトリ構成)
- [書籍の章構成](#書籍の章構成)
- [クイックスタート](#クイックスタート)

## リポジトリ構成

```

```

## 書籍の章構成

| 章 | タイトル | ノートブック |
|:--:|:--------|:------------|
| 第1章 | 大規模言語モデルの歴史と本書で得られること | — |
| 第2章 | Transformerモデルの作成 | [`notebooks/chapter02/`](notebooks/chapter02/) |
| 第3章 | GPTモデルの作成 | [`notebooks/chapter03/`](notebooks/chapter03/) |
| 第4章 | 大規模言語モデルの学習 | [`notebooks/chapter04/`](notebooks/chapter04/) |
| 第5章 | アラインメント | [`notebooks/chapter05/`](notebooks/chapter05/) |
| 第6章 | 推論モデル | — |
| 付録 | NumPy、PyTorch入門 | — |



## クイックスタート

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

### サンプルの実行

```bash
# 夏目漱石コーパスでGPTを学習
uv run python examples/train_gpt_soseki.py
```


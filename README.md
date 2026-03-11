# LLM from Scratch

大規模言語モデル（LLM）の基礎から実装までを学ぶためのリポジトリです。TransformerアーキテクチャからGPTモデルの実装、分散学習、アラインメントまでを段階的に理解できます。

## セットアップ

```bash
git clone https://github.com/elith-co-jp/book-llm-from-scratch.git
cd book-llm-from-scratch

# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

## ノートブック

各章のノートブックはGoogle Colabで直接開いて実行できます。

### 2章: Transformerの実装

| セクション | 内容 | Colab |
|:--|:--|:--:|
| 2.2 | アテンション機構 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter02/section2.ipynb) |
| 2.3 | アテンション以外の部品 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter02/section3.ipynb) |
| 2.4 | Transformerを作る | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter02/section4.ipynb) |
| 2.5 | Transformerの学習と推論 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter02/section5.ipynb) |

### 3章: GPTモデルの学習

| セクション | 内容 | Colab |
|:--|:--|:--:|
| 3.3 | 夏目漱石テキストでGPT訓練 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter03/train_gpt_soseki.ipynb) |

### 4章: 大規模化と分散学習

| セクション | 内容 | Colab |
|:--|:--|:--:|
| 4.1 | データセット前処理 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter04/section01_dataset_preprocessing.ipynb) |
| 4.2 | データ並列 / テンソル並列 | `.py`（torchrun で実行） |
| 4.3 | GPT-2 事前学習 | `.py`（torchrun / deepspeed で実行） |
| 4.4 | LoRA ファインチューニング | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter04/section04_lora.ipynb) |

### 5章: アラインメント

| セクション | 内容 | Colab |
|:--|:--|:--:|
| 5.2 | インストラクションチューニング | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter05/section2.ipynb) |
| 5.3 | DPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elith-co-jp/book-llm-from-scratch/blob/main/notebooks/chapter05/section3.ipynb) |

## リポジトリ構成

```
notebooks/              # 各章のノートブック・スクリプト
llm_from_scratch/       # Pythonパッケージ（2章ノートブックから参照）
├── transformer/        #   Transformerモジュール
└── gpt/                #   GPTモジュール
refs/
└── errors.md           # 正誤表
```

**ファイル形式の使い分け:**
- `.ipynb`: 対話的に実行可能なノートブック（Google Colabでも実行可）
- `.py`: `torchrun` / `deepspeed` で起動する分散学習スクリプト

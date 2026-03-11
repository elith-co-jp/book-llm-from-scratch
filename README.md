# LLM from Scratch

<table>
<tr>
<td width="200">
<a href="https://www.amazon.co.jp/dp/4296205250/">
<img src="refs/cover.jpg" alt="作ってわかる大規模言語モデルの仕組み" width="180">
</a>
</td>
<td>

本リポジトリは書籍『[**作ってわかる大規模言語モデルの仕組み**](https://www.amazon.co.jp/dp/4296205250/)』（日経BP、2026年）の公式サポートリポジトリです。

大規模言語モデル（LLM）の基礎から実装までを、Transformerアーキテクチャ → GPTモデル → 事前学習 → アラインメントと段階的に学ぶことができます。

本書の正誤訂正情報は[こちら](refs/errors.md)をご覧ください。

</td>
</tr>
</table>

---

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

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

## リポジトリ構成

```
notebooks/
├── chapter02/          # 2章: Transformerの実装
│   ├── section2.ipynb  #   アテンション機構
│   ├── section3.ipynb  #   アテンション以外の部品
│   ├── section4.ipynb  #   Transformerを作る
│   └── section5.ipynb  #   Transformerの学習と推論
├── chapter03/          # 3章: GPTモデルの学習
│   └── train_gpt_soseki.ipynb  # 夏目漱石テキストでGPT訓練（Colab対応）
├── chapter04/          # 4章: 大規模化と分散学習
│   ├── section01_dataset_preprocessing.ipynb  # データセット前処理
│   ├── section02_data_parallel.py             # データ並列（torchrun）
│   ├── section02_tensor_parallel.py           # テンソル並列（torchrun）
│   ├── section03_train_gpt2.py                # GPT-2事前学習（torchrun/deepspeed）
│   └── section04_lora.ipynb                   # LoRAファインチューニング
└── chapter05/          # 5章: アラインメント
    ├── section2.ipynb  #   インストラクションチューニング
    └── section3.ipynb  #   DPO

llm_from_scratch/       # Pythonパッケージ（2章ノートブックから参照）
├── transformer/        #   Transformerモジュール
└── gpt/                #   GPTモジュール

refs/
└── errors.md           # 正誤表
```

**ファイル形式の使い分け:**
- `.ipynb`: 対話的に実行可能なノートブック（Google Colabでも実行可）
- `.py`: `torchrun` / `deepspeed` で起動する分散学習スクリプト

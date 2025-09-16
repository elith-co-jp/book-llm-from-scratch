# LLM from Scratch

大規模言語モデル（LLM）の基礎から実装までを学ぶためのリポジトリです。TransformerアーキテクチャからGPTモデルの実装まで、段階的に理解を深めることができます。

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

詳細な実行例は `examples/train_gpt.py` を参照してください。

```bash
# サンプルスクリプトの実行
uv run python examples/train_gpt.py
```
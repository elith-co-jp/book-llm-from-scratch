　それでは本節（3.2節）から、2章で実装したTransformerモデルを発展させる形で、初期のGPTモデルであるGPT-2を実装していきます。GPT-2は、モデルのアーキテクチャや学習方法の詳細が論文で公開されていて、重みも公開されているオープンウェイトモデルです。

　最初に本節で「トークナイザ（Tokenizer）」を実装します。自然言語処理において、言語モデルは人間が使用する言葉をそのまま理解しているように見えるかもしれませんが、実際には背後で複雑な処理が行われています。人間の言葉を機械が理解しやすい形に変換するための重要なステップがトークナイザによるトークン化です。

# 3.2.1 トークナイザの概要

　本項（3.2.1項）ではまず、トークナイザの役割とその実装方法について詳しく解説します。

＃＃小見出し

### **トークナイザの役割**

　2.3節でも説明したように、LLMはその内部で人間が用いる言葉を分割し、それぞれを数値に変換して処理しています。このような処理を行うツールや手法を、トークナイザと呼びます▲脚注番号▲。

＃＃脚注
参考文献：Summary of the tokenizers、HuggingFace、https://huggingface.co/docs/transformers/tokenizer_summary

＃＃

　トークナイザは、自然言語処理タスクにおいて、以下の重要な役割を果たします。

- テキストを単語や部分文字列（トークン）に分割する処理を行います。
- GPTモデルが理解しやすい形式に入力テキストを変換します。
- トークン化の方法には、単語単位、文字単位、BPE（Byte Pair Encoding）などがあります。

　以下は、単語単位でトークン化をする簡単な例です。

```python
import re

def tokenize_text(text):
    # 正規表現を使って、単語とスペース・句読点に分割
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

text = "I like apples. You like oranges, not apples."
tokens = tokenize_text(text)
print(tokens)
```

出力:

```python
['I', 'like', 'apples', '.', 'You', 'like', 'oranges', ',', 'not', 'apples', '.']
```

　「`tokens = re.findall(r'\w+|[^\w\s]', text)`」の部分では、正規表現を使ってテキストを単語と句読点に分割しています。「`\w+|[^\w\s]`」の部分が正規表現です。慣れないと呪文のようで難解に感じるかもしれません。正規表現は、文字列の検索、置換、抽出などの文字列処理で広く利用されています。また、多くのプログラミング言語やテキストエディタ、コマンドラインツールなどでも正規表現がサポートされており、柔軟かつ強力なパターンマッチングが可能です。

＃＃小見出し

### **トークン化の手順**

　トークン化の一般的な手順は以下の通りです。

- 入力テキストの正規化：大文字・小文字の統一や句読点の処理などを行います。
- 単語や部分文字列へのトークン化：正規化されたテキストを、単語や部分文字列に分割します。
- トークンIDへの変換：分割されたトークンに対して、一意の整数IDを割り当てます。

　以下は、トークン化の手順を示すPythonコードの例です。

```python
def preprocess_text(text):
    # 小文字に変換
    text = text.lower()
    # 句読点の前後にスペースを追加
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    return text

def tokenize_text(text):
    # 正規表現を使って、単語とスペース・句読点に分割
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def assign_token_ids(tokens, vocab):
    token_ids = []
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            token_ids.append(vocab['<UNK>'])  # 未知語の処理
    return token_ids

text = "I like apples. You like oranges, not apples."
preprocessed_text = preprocess_text(text)
tokens = tokenize_text(preprocessed_text)
print("Tokens:", tokens)

vocab = {'<PAD>': 0, '<UNK>': 1, 'i': 2, 'like': 3, 'apples': 4, '.': 5, 'you': 6, 'oranges': 7, ',': 8, 'not': 9}
token_ids = assign_token_ids(tokens, vocab)
print("Token IDs:", token_ids)
```

出力:

```python
Tokens: ['i', 'like', 'apples', '.', 'you', 'like', 'oranges', ',', 'not', 'apples', '.']
Token IDs: [2, 3, 4, 5, 6, 3, 7, 8, 9, 4, 5]
```

　このコードは、テキストデータの前処理とトークン化を行う基本的な手順を示しています。

　`preprocess_text`関数では、入力テキストの正規化をします。テキストを小文字に変換し、句読点の前後にスペースを追加します。これにより、後続の処理が容易になります。

　`tokenize_text`関数は、単語へのトークン化をします。前処理されたテキストを単語や句読点などの個別の「トークン」に分割します。

　`assign_token_ids`関数は、トークンIDに変換します。予め定義されたボキャブラリーを使用して、各トークンに対応する数値IDを割り当てます。

　最後に、サンプルテキストに対してこれらの関数を順に適用し、得られたトークンとトークンIDを出力しています。この結果、テキストが機械学習モデルで扱いやすい数値形式に変換されます。

＃＃小見出し

### **ボキャブラリー**

　ボキャブラリーは、トークンとトークンIDの対応関係を定義します。ボキャブラリーのサイズ（トークンの種類数）がモデルの表現力に影響を与えます。ボキャブラリーは、モデルの学習時に構築され、推論時にも使用されます。未知語（Out-of-Vocabulary, OOV）の処理方法もボキャブラリーの設計に関連します。

　以下は、ボキャブラリーを構築するPythonコードの例です。

```python
from collections import Counter

def build_vocabulary(tokens, max_size=None, min_freq=1, special_tokens=None):
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>']

    # トークンの出現頻度を計算
    token_counts = Counter(tokens)

    # 出現頻度の高い順にトークンをソート
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # 最小出現頻度と最大ボキャブラリーサイズでボキャブラリーを制限
    filtered_tokens = [token for token, freq in sorted_tokens if freq >= min_freq][:max_size]

    # 特殊トークンを追加
    vocabulary = {token: idx for idx, token in enumerate(special_tokens + filtered_tokens)}
    return vocabulary

# トークンのリスト
tokens = ['i', 'like', 'apples', '.', 'you', 'like', 'oranges', ',', 'not', 'apples', '.']

# ボキャブラリーの構築
vocab = build_vocabulary(tokens, max_size=7, min_freq=1)
print("Vocabulary:", vocab)
```

出力:

```python
Vocabulary: {'<PAD>': 0, '<UNK>': 1, 'like': 2, 'apples': 3, '.': 4, 'i': 5, 'you': 6, 'oranges': 7, ',': 8}
```

　この例では、`build_vocabulary`関数を使ってボキャブラリーを構築しています。特殊トークン（`<PAD>`と`<UNK>`）を追加し、出現頻度の高い順にトークンをソートしています。最小出現頻度（`min_freq`）と最大ボキャブラリーサイズ（`max_size`）でボキャブラリーを制限しています。

　適切なトークン化手法とボキャブラリーの設計により、GPTモデルの学習効率と推論精度を向上させることができます。

＃＃小見出し

### まとめ

　本項では、トークナイザの概要とその重要性について学びました。トークナイザは、テキストを単語や部分文字列に分割し、モデルが理解しやすい形式に変換する役割を果たします。具体的には、単語単位や文字単位、BPEなどのトークン化方法があり、それぞれに特有の利点があります。また、トークン化の手順やボキャブラリーの構築方法も紹介しました。適切なトークン化とボキャブラリーの設計により、モデルの学習効率と精度が大きく向上します。これらの基礎知識を基に、次項ではより高度なトークナイザの設計と応用について探ります。

# 3.2.2 トークナイザの入力処理

　トークナイザは、自然言語をモデルが理解できる形式に変換する重要なツールです。その中でも、特殊トークンの追加と最大入力長の設定は、モデルの精度と効率に大きな影響を与える要素です。本項では、これらのトピックについて詳しく解説します。特殊トークンの役割や追加方法、そして最大入力長に応じたトークンの切り詰めについて学びます。

＃＃小見出し

### **特殊トークン**

　特殊トークンは、モデルの入力形式や学習タスクに応じて使用される特別なトークンです。以下は代表的な特殊トークンとその役割です。

- [CLS]（Classification Token）：文の先頭に追加され、文全体の表現を学習するために使用されます。
- [SEP]（Separator Token）：文の終わりや、質問と回答のペアを分割するために使用されます。
- [MASK]（Masked Token）：マスク言語モデル（MLM）の学習において、予測対象のトークンを示すために使用されます。
- [PAD]（Padding Token）：バッチ処理の際に、シーケンス長をそろえるために使用されます。
- [UNK]（Unknown Token）：ボキャブラリーにない未知の単語を表すために使用されます。

　以下は、特殊トークンを追加する例です。

```python
special_tokens = ['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']
vocab = {'[CLS]': 0, '[SEP]': 1, '[MASK]': 2, '[PAD]': 3, '[UNK]': 4, 'i': 5, 'like': 6, 'apples': 7, '.': 8}

def add_special_tokens(tokens, max_length):
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    padding_length = max_length - len(tokens)
    tokens += ['[PAD]'] * padding_length
    return tokens

tokens = ['i', 'like', 'apples', '.']
max_length = 10
tokens_with_special = add_special_tokens(tokens, max_length)
print("Tokens with special tokens:", tokens_with_special)

```

出力:

```
Tokens with special tokens: ['[CLS]', 'i', 'like', 'apples', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
```

＃＃小見出し

### **最大入力長（シーケンス長）**

　最大入力長は、GPTモデルが一度に処理できるトークンの最大数を示します。これは、モデルのアーキテクチャに依存し、通常は512や1024などの値が使用されます。入力テキストがこの最大長を超える場合、トークンの切り詰めが必要になります。

　以下は、最大入力長に合わせてトークンを切り詰める例です。

```python
def truncate_tokens(tokens, max_length):
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + ['[SEP]']
    return tokens

tokens = ['[CLS]', 'i', 'like', 'apples', '.', 'you', 'like', 'oranges', ',', 'not', 'apples', '.', '[SEP]']
max_length = 10
truncated_tokens = truncate_tokens(tokens, max_length)
print("Truncated tokens:", truncated_tokens)

```

出力:

```
Truncated tokens: ['[CLS]', 'i', 'like', 'apples', '.', 'you', 'like', 'oranges', ',', '[SEP]']
```

　最大入力長を適切に設定することで、GPTモデルが効率的に入力を処理できるようになります。ただし、入力テキストが長すぎる場合は、切り詰めによって情報が失われる可能性があるため、注意が必要です。

＃＃小見出し

### まとめ

　本項では、トークナイザの入力処理における特殊トークンの重要性と最大入力長の設定について学びました。特殊トークンは、モデルの入力形式や学習タスクに応じて使用され、特定の役割を果たします。また、最大入力長を設定することで、モデルが一度に処理できるトークン数を制御し、効率的な入力処理が可能となります。適切な特殊トークンの追加と最大入力長の管理により、モデルの性能を最大限に引き出すことができます。これらの知識を基に、次項ではさらに高度なトークナイザの最適化方法について探ります。

# 3.2.3 埋め込み

　本項では、2章で解説したTransformerの要素技術が、GPTでどのように活用・変更されているかを説明しました。

　トークンIDは、そのままではGPTモデルで処理するのに適していません。トークンIDを高次元のベクトルに変換する仕組みが必要です。それが2.3節でも説明した埋め込みです。埋め込みは、トークンの意味や関係性を表現するためのベクトルであり、モデルが自然言語を理解するために重要な役割を果たします。

　GPTでは、入力埋め込みは2章と同じ仕組みですが、位置エンコーディングについては重要な違いがあります。以下でそれぞれ詳しく見ていきましょう。

＃＃小見出し

### 入力埋め込み

　入力埋め込みは、2.3節で実装した「トークン埋め込み」と同じ仕組みです。2章で説明したように、トークンIDは離散的な値であり、勾配計算や意味的な類似性の表現には適していないため、連続的なベクトル空間に埋め込む必要があります。このベクトル表現により、単語の意味や単語同士の関係性を捉えられるように学習されます。

　GPTでも2章と同様に`nn.Embedding`を使用して実装します。違いは、GPT特有のボキャブラリーサイズ（例：GPT-2では50,257）と埋め込み次元（例：768次元）を使用する点です。この学習は、GPTモデルの他の部分の学習と同時に行われ、モデルが自然言語を理解するために最適化されていきます。

　以下は、PyTorchを使用して入力埋め込みを実装する例です。

```python
import torch
import torch.nn as nn

vocab_size = 10
embedding_dim = 128

# 入力埋め込み層の定義
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# トークンIDのバッチ
token_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

# 入力埋め込みの計算
input_embeddings = embedding_layer(token_ids)
print("Input embeddings shape:", input_embeddings.shape)

```

出力:

```
Input embeddings shape: torch.Size([2, 5, 128])
```

　この例では、`nn.Embedding`を使用して入力埋め込み層を定義しています。`vocab_size`はボキャブラリーサイズ、`embedding_dim`は埋め込みベクトルの次元数です。トークンIDのバッチ（`token_ids`）を入力として与えると、対応する入力埋め込みが計算されます。

　また、実際に利用する際にはこの埋め込み層を含むモデル全体を学習データを用いて訓練する必要があります。訓練を通じて、埋め込み層のパラメータが最適化され、単語の意味や関係性を適切に表現できるようになります。

＃＃小見出し

### 位置埋め込み

　2章でも説明したように、Transformerベースのモデルは入力トークンの位置関係を直接的には理解できません。例えば、以下の2つの文を考えてみましょう。

1. "I love cats and dogs."
2. "And love cats I dogs."

　位置情報がなければ、これらの文は同じトークンで構成されているため、全く同じ入力として扱われてしまいます。この問題を解決するため、位置情報を表現する仕組みが必要になります。

　2.3節では、Transformerの原論文に基づく**正弦波位置エンコーディング（Sinusoidal Positional Encoding）**を実装しました。これは固定的な数学的関数（sinとcos）を使用して位置情報を表現する方法でした。

　一方、GPTでは**学習可能な位置埋め込み（Learnable Position Embedding）**を採用しています。これは各位置に対して学習可能なパラメータベクトルを用意する方式で、以下の違いがあります。

- **正弦波方式**：固定的な数学関数で計算、学習不要、任意の長さに対応可能
- **学習可能方式**：`nn.Embedding`で実装、学習により最適化、最大長の事前設定が必要

　GPTがこの方式を採用する理由は、データから位置パターンを学習することで、タスクに特化したより良い表現を獲得できるためです。位置埋め込みを入力埋め込みに加算することで、GPTモデルは各トークンの位置を区別し、文法的に正しい文章を理解・生成できるようになります。

　なお、位置埋め込みには、絶対位置埋め込みと相対位置埋め込みの2種類があります。絶対位置埋め込みは、シーケンス内の各トークンの絶対的な位置を表現します。相対位置埋め込みは、トークン間の相対的な位置関係を表現します。GPT-2では前者を採用しています。

　以下は、PyTorchを使用してGPTの絶対位置埋め込みを実装する例です。

```python
import torch
import torch.nn as nn

seq_length = 5
embedding_dim = 128

# 位置埋め込み層の定義
position_embedding_layer = nn.Embedding(seq_length, embedding_dim)

# 位置IDの生成
position_ids = torch.arange(seq_length).unsqueeze(0)

# 位置埋め込みの計算
position_embeddings = position_embedding_layer(position_ids)
print("Position embeddings shape:", position_embeddings.shape)

```

出力:

```
Position embeddings shape: torch.Size([1, 5, 128])

```

　この例では、`nn.Embedding`を使用して位置埋め込み層を定義しています。`seq_length`はシーケンスの長さ、`embedding_dim`は埋め込みベクトルの次元数です。位置IDを生成し、位置埋め込み層に入力することで、対応する位置埋め込みが計算されます。

　入力埋め込みと位置埋め込みを組み合わせることで、GPTモデルは入力トークンの意味と位置の両方を考慮しながらテキストを処理できるようになります。

### まとめ

　本項では、埋め込みの基本的な概念とその重要性について学びました。トークンIDを高次元ベクトルに変換する入力埋め込みは、モデルがテキストの意味を理解するための基盤です。また、トークンの位置情報を提供する位置埋め込みは、文の構造を正しく理解するために不可欠です。これらの埋め込みを組み合わせることで、GPTモデルはより精度の高い自然言語処理を実現します。次項では、これらの埋め込みを活用した具体的なモデルのトレーニング方法について探っていきます。

# 3.2.4 トークナイザの応用

　これまでに、トークナイザの基本的な概念とその役割について学びました。自然言語をサブワードに分割することで、言語モデルがより効果的にテキストを理解し、処理することが可能となります。本項では、特にサブワードベースのトークナイザであるBPEを中心に詳しく説明します。その他、WordPiece、SentencePieceについても簡単に解説します。これらのトークナイザの仕組みと応用について理解を深めていきます。

＃＃小見出し

### Byte-Pair Encoding（BPE）アルゴリズムの詳細解説

　Byte-Pair Encoding（BPE）は、元々データ圧縮のために考案されたアルゴリズムですが、現代の自然言語処理（NLP）では非常に効果的なトークン化手法として利用されています。特に、GPT-2やRoBERTaのような大規模言語モデルにおいて、サブワードレベルでの柔軟なトークン分割を可能にする点で重要な役割を果たしています。

＃＃小見出し

### BPEの理論的背景と学習プロセス

　BPEは、テキストデータをサブワードや文字単位に分解し、最適なトークンセットを生成することを目的としています。この方法により、未知の単語や造語にも対応できる汎用的なモデルを構築することが可能です。BPEの学習プロセスは以下です。

1. **初期ボキャブラリーの生成**: テキストコーパス内の全単語を文字単位に分解し、これを初期ボキャブラリーとします
2. **頻度に基づく結合**: 文字ペアの出現頻度を計算し、最も頻度の高いペアを結合して新たなトークンを作成します。この操作は、目標とするボキャブラリーサイズに達するまで繰り返されます
3. **反復的マージプロセス**: 作成された新しいトークンを用いてさらにペアを結合し、最終的に頻度に基づいた最適なサブワードセットを構築します

＃＃小見出し

### トークン化のプロセス

　BPEを用いたトークン化は以下の手順です。

1. **プレトークン化**: テキストを単語やサブワードに分割し、空白、句読点、特殊文字などを適切に処理します
2. **文字の分割と結合ルールの適用**: テキストを個々の文字に分割し、学習された結合ルールを順次適用してトークンを生成します

＃＃小見出し

### PythonによるBPEトークナイザの実装と結果

　ここでは、シンプルなBPEトークナイザをPythonで実装し、各ステップの出力結果を示します。難しそうに思えますが、GPT-2で利用されているものなので頑張って読み進めてください。

**1. コーパスの準備**

　まず、処理対象となるテキストデータを用意します。ここでは、本節に関係のある文書を用意しました。

```python
corpus = [
    "Large language models are transforming the landscape of natural language processing.",
    "Understanding tokenization is crucial for building efficient models.", 
    "This chapter explores the intricacies of BPE and its impact on LLM performance.",
    "By mastering these techniques, you can enhance the capabilities of your models.",
    "Each token in a sequence carries semantic meaning and contextual information.",
    "Modern tokenizers split text into meaningful subword units called tokens.",
    "A well-designed token vocabulary improves model efficiency and accuracy.",
    "Token-level representations enable better handling of rare and unseen words.",
    "Tokenization strategies directly affect how models process and understand text.",
    "The choice of tokenization method influences both training speed and final performance.",
    "Effective token boundaries help preserve linguistic structures in the data.",
    "Language models and LLMs benefit from sophisticated tokenization approaches.",
]

```

**2. シンプルなトークナイザの実装と出力**

　単語を空白で区切るシンプルなトークナイザを作成します。

```python
def simple_tokenizer(text):
    tokens = text.split()
    return tokens

# 出力（コーパスの1行目）
tokens = simple_tokenizer(corpus[0])
print(tokens)

```

**出力:**

```python
['Large', 'language', 'models', 'are', 'transforming', 'the', 'landscape', 'of', 'natural', 'language', 'processing.']

```

**解説:**

　このプログラムは、文を単語単位で分割します。結果として、各単語がトークンとしてリストに格納されます。

**3. 単語頻度の計算と出力**

　コーパス内の各単語の出現頻度を計算します。

```python
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    tokens = simple_tokenizer(text)
    for token in tokens:
        word_freqs[token] += 1

print(word_freqs)

```

**出力:**

```python
defaultdict(<class 'int'>, {
    'Large': 1, 'language': 2, 'models': 1, 'are': 1, 'transforming': 1, 
    'the': 3, 'landscape': 1, 'of': 3, 'natural': 1, 'processing.': 1,
    'Understanding': 1, 'tokenization': 1, 'is': 1, 'crucial': 1, 'for': 1, 
    'building': 1, 'efficient': 1, 'models.': 2, 'This': 1, 'chapter': 1, 
    'explores': 1, 'intricacies': 1, 'BPE': 1, 'and': 1, 'its': 1, 
    'impact': 1, 'on': 1, 'LLM': 1, 'performance.': 1, 'By': 1, 
    'mastering': 1, 'these': 1, 'techniques,': 1, 'you': 1, 'can': 1, 
    'enhance': 1, 'capabilities': 1, 'your': 1
})

defaultdict(<class 'int'>, {
    'Large': 1, 'language': 2, 'models': 3, 'are': 1, 'transforming': 1, 
    'the': 4, 'landscape': 1, 'of': 5, 'natural': 1, 'processing.': 1, 
    'Understanding': 1, 'tokenization': 3, 'is': 1, 'crucial': 1, 'for': 1, 
    'building': 1, 'efficient': 1, 'models.': 2, 'This': 1, 'chapter': 1, 
    ...
})
```

**解説:**

　この出力では、各単語の出現回数が記録されています。同じ単語が複数回現れる場合、その頻度が増加しています。

**4. 初期ボキャブラリーの作成と出力**

　コーパス内のすべての文字を使って初期ボキャブラリーを作成します。

```python
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)
```

**出力:**

```python
[',', '-', '.', 'A', 'B', 'E', 'L', 'M', 'P', 'T', 'U', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
```

**解説:**

　この出力は、コーパス内のすべての文字をアルファベット順に並べたものです。ここで、空白や句読点も含まれています。

**5. 単語の文字分割と出力**

　各単語を構成文字に分解し、これを辞書に格納します。

```python
splits = {word: [c for c in word] for word in word_freqs.keys()}
print(splits)

```

**出力:**

```python
{
    'Large': ['L', 'a', 'r', 'g', 'e'], 'language': ['l', 'a', 'n', 'g', 'u', 'a', 'g', 'e'],
    'models': ['m', 'o', 'd', 'e', 'l', 's'], 'are': ['a', 'r', 'e'], 'transforming': ['t', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'i', 'n', 'g'],
    ...
}
```

**解説:**

　各単語がその構成文字に分解され、リストに格納されています。この分解により、後の結合処理が容易になります。

**6. 文字ペアの頻度計算と出力**

　各単語内の文字ペアの出現頻度を計算します。

```python
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)
print(pair_freqs)
```

**出力:**

```python
defaultdict(<class 'int'>, {
    ('L', 'a'): 2, ('a', 'r'): 6, ('r', 'g'): 1, ('g', 'e'): 4, 
    ('l', 'a'): 4, ('a', 'n'): 22, ('n', 'g'): 13, ('g', 'u'): 4,
    ...
})
```

**解説:**

　この出力では、各単語内の文字ペアの出現頻度が計算されています。この頻度に基づいて、次の結合ステップが行われます。

**7. 最も頻度の高いペアの結合**

頻度が最も高い文字ペアを特定し、それを結合して新しいトークンを作成します。

```python
best_pair = max(pair_freqs, key=pair_freqs.get)
print(best_pair)
```

**出力:**

```python
('a', 'n')
```

**解説:**

　ここで、最も頻度の高い文字ペア `('a', 'n')` が特定されました。このペアが次のステップで結合されます。

**8. 結合ペアの適用と出力**

　特定したペアをすべての単語に対して適用し、新しいトークンを作成します。

```python
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

splits = merge_pair(*best_pair, splits)
print(splits)

```

**出力:**

```python
{
    'Large': ['L', 'a', 'r', 'g', 'e'], 
    'language': ['l', 'an', 'g', 'u', 'a', 'g', 'e'], 
    'models': ['m', 'o', 'd', 'e', 'l', 's'],
    ...
}
```

**解説:**

　この出力では、特定したペア `('a', 'n')` が結合され、`'an'` という新しいトークンが作成されました。このプロセスは、ボキャブラリーサイズが目標に達するまで繰り返されます。

**9. ボキャブラリーサイズの設定とトレーニングループの出力**

　目標とするボキャブラリーサイズに達するまで、ペアの結合を繰り返します。

```python
num_merges = 100  # 結合回数を10回に設定
vocab = [""] + alphabet.copy()
merges = {}

for _ in range(num_merges):
    pair_freqs = compute_pair_freqs(splits)
    best_pair = max(pair_freqs, key=pair_freqs.get)
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

print(vocab)

```

**出力:**

```python
['', ',', '-', '.', 'A', 'B', 'E', 'L', 'M', 'P', 'T', 'U', 'a', 'b', 'c', 'd', 
'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
'v', 'w', 'x', 'y', 'z', 'en', 'in', 'es', 'and', 'ti', 'ing', 'er', 'ok', 
'oken', 'el', 'od', 'or', 'token', 'on', 'th', 'ca', 'mod', 'model', 'pr', 
...
'mean', 'meaning', 'wor', 'word', 's.', 'ed', 'cy', 'Token', 're', 'pres']
```

**解説:**

　この出力は、100回の結合操作を通じて構築されたボキャブラリーです。`'en'`, `'in'`, `'es'` などの新しいトークンがボキャブラリーに追加されています。ここで通常のBPEアルゴリズムであれば、ボキャブラリーサイズの数字を追加して、そのボキャブラリーサイズになるまでループを回しますが、ここでは理解のしやすさのため、決まったループの回数を回すようにしています。

**10. 新規テキストのトークン化と出力**

　学習したルールを用いて新しいテキストをトークン化します。

```python
def bpe(text):
    tokens = simple_tokenizer(text)
    splits = [list(word) for word in tokens]  # 各単語を文字に分割

    for pair, merge in merges.items():
        for split in splits:
            merged_split = []
            i = 0
            while i < len(split):
                is_pair = (
                    i < len(split) - 1 and  # 現在のインデックスがリストの最後ではない
                    split[i] == pair[0] and  # 現在の文字がペアの最初の文字と一致
                    split[i + 1] == pair[1]  # 次の文字がペアの2番目の文字と一致
                )
                if is_pair:
                    merged_split.append(merge)  # 結合する
                    i += 2  # 2文字分をスキップ
                else:
                    merged_split.append(split[i])
                    i += 1
            split[:] = merged_split  # 結果をsplitに反映

    return [token for split in splits for token in split]

# テキストのトークン化
result = bpe("Tokenization improves LLMs.")
print(result)

```

**出力:**

```python
['Token', 'iz', 'ation', 'im', 'pro', 'v', 'es', 'LLM', 's.']
```

**解説:**

　この短い文章でも、「Tokenizing」や「improves」といった単語がBPEによってサブワードに分割されている様子がわかります。例えば「Tokenization」は「Token」「iz」「ation」に分割され、これによりLLMがより効果的に単語を処理できるようになります。

**11. プログラムのまとめ**

　1〜10のプログラムをまとめると以下のように表されます。難しそうに思えたBPEですが、実装してみると非常にシンプルであることが理解できます。

```python
from collections import defaultdict

# 1. コーパスの準備
corpus = [
    "Large language models are transforming the landscape of natural language processing.",
    "Understanding tokenization is crucial for building efficient models.", 
    ...
]

# 2. シンプルなトークナイザの実装
def simple_tokenizer(text):
    tokens = text.split()
    return tokens

# 3. 単語頻度の計算
word_freqs = defaultdict(int)
for text in corpus:
    tokens = simple_tokenizer(text)
    for token in tokens:
        word_freqs[token] += 1

# 4. ベースボキャブラリーの作成
alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# 5. 単語の文字分割
splits = {word: [c for c in word] for word in word_freqs.keys()}

# 6. 文字ペアの頻度計算
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

# 7. 最も頻度の高いペアの結合
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

# 8. ボキャブラリーサイズの設定とトレーニングループ
num_merges = 10  # 結合回数を10回に固定
vocab = [""] + alphabet.copy()
merges = {}

for _ in range(num_merges):
    pair_freqs = compute_pair_freqs(splits)
    best_pair = max(pair_freqs, key=pair_freqs.get)
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

# 9. 新規テキストのトークン化
def bpe(text):
    tokens = simple_tokenizer(text)
    splits = [list(word) for word in tokens]  # 各単語を文字に分割

    for pair, merge in merges.items():
        for split in splits:
            merged_split = []
            i = 0
            while i < len(split):
                is_pair = (
                    i < len(split) - 1 and  # 現在のインデックスがリストの最後ではない
                    split[i] == pair[0] and  # 現在の文字がペアの最初の文字と一致
                    split[i + 1] == pair[1]  # 次の文字がペアの2番目の文字と一致
                )
                if is_pair:
                    merged_split.append(merge)  # 結合する
                    i += 2  # 2文字分をスキップ
                else:
                    merged_split.append(split[i])
                    i += 1
            split[:] = merged_split  # 結果をsplitに反映

    return [token for split in splits for token in split]

# 10. テキストのトークン化
result = bpe("Tokenization improves LLMs.")
print(result)
```

＃＃小見出し

### その他のトークナイザ

　ここでその他のトークナイザで有名な**WordPiece**と**SentencePiece**について紹介します。

**WordPiece**

　WordPieceは、BPEと同様にサブワード分割のアルゴリズムですが、その動作原理は異なります。BPEが出現頻度に基づいてサブワードを生成するのに対し、WordPieceは確率に基づいてサブワードを作成します。
　WordPieceは確率最大化に基づくサブワード分割を行います。具体的には、コーパス全体の尤度を最大化するようにマージを選択します。例えば、"un"と"likely"を"unlikely"にマージする際、P(unlikely) > P(un) × P(likely)となる場合にマージを実行します。この条件は、結合後の確率が結合前の独立した確率の積よりも高いことを意味し、より自然な単語の分割を実現します。

　この確率ベースのアプローチにより、WordPieceはボキャブラリーサイズを最小化しつつ、より意味的に関連性の高いサブワードを生成することができます。BPEが単純な頻度でマージを決定するのに対し、WordPieceは言語モデルの観点から最適な分割を選択するため、単なる文字の並びではなく、実際の言語における意味のまとまり（例：「un-」という否定の接頭辞）を適切に保持できます。

　WordPieceの利点の一つは、未知語に対する柔軟性です。未知語を構成するサブワードを個別のトークンとして扱えるため、新たに出現する単語や特殊な用語も適切に処理することが可能です。この特性は、特に大規模なテキストデータセットを扱う際に有効で、新しい単語に対するモデルの適応性を高めます。

　WordPieceは、BERTなどの言語モデルで使用されています。BERTをはじめとするこれらのモデルでは、WordPieceによって生成されたサブワードのトークンが、テキストの理解と生成において重要な役割を果たしています。このアルゴリズムによるサブワードへの細かい分割は、モデルがより複雑な文脈やニュアンスを捉えるのに役立ち、結果として言語処理の精度を向上させます。

**SentencePiece**

　SentencePieceは、言語に依存しない統一的なトークン化を実現するアルゴリズムです。BPEやWordPieceが単語境界（スペース）を前提とするのに対し、SentencePieceは生のテキストを直接処理できる点が最大の特徴です。

　SentencePieceの動作原理は、BPEまたはUnigram言語モデルのいずれかを選択して使用できますが、主流はUnigram言語モデルベースの手法です。このアプローチでは、まず大量の候補サブワードを用意し、そこから最適なボキャブラリーを選別していきます。具体的には、削除しても文章の生成確率への影響が小さいサブワードから順に削除し、設定したボキャブラリーサイズまで絞り込みます。

　SentencePieceの最大の利点は、**言語非依存性**です。英語のようにスペースで単語を区切る言語も、日本語や中国語のようにスペースを使わない言語も、同じ方法で処理できます。例えば、「私は猫が好きです」という日本語文を、前処理なしに直接「私」「は」「猫」「が」「好き」「です」のようなサブワードに分割できます。これは3.3節で扱う夏目漱石のテキストでGPTを学習する際にも活用します。

　もう一つの重要な特徴は、**特殊文字の扱い**です。SentencePieceはスペースを「▁」（アンダースコア）という特殊文字で表現し、トークンの境界を明確に保持します。これにより、トークン化後も元のテキストを完全に復元可能となります。

　SentencePieceは、T5、mBERT、XLM-RoBERTaなどの多言語モデルで広く採用されています。特に、複数の言語を一つのモデルで扱う場合や、日本語のような「非分かち書き言語」を処理する場合に威力を発揮します。GoogleのSentencePieceライブラリはC++で実装されており、Pythonバインディングも提供されているため、実用的な処理速度で利用可能です。

＃＃小見出し

### まとめ

　本項では、サブワードベースのトークナイザであるBPE、WordPiece、SentencePieceについて解説しました。BPEは出現頻度に基づいてサブワードを生成し、未知語に柔軟に対応できるアルゴリズムです。WordPieceは確率に基づいてサブワードを作成し、テキストのセマンティックな側面を重視します。SentencePieceは言語の特性に依存せず、Unigram言語モデルを用いてサブワードを最適化するアプローチを取ります。これらのトークナイザを適切に活用することで、言語モデルの精度と汎用性を向上させることができます。次節では、これらのトークナイザを具体的な自然言語処理タスクに応用する方法について探っていきます。
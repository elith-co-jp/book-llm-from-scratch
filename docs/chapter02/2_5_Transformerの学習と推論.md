# 2.5 Transformer の学習と推論

2.4 節では Transformer モデルと、適切に学習できた場合の文章生成メソッドの実装を行いました。この節では Transformer を学習する方法を説明します。ソースコードとしては実データを用いた学習ができる部分まで紹介し、実際に出力を確認します。その後に発展的な内容として、2.4 節で紹介した以外の文章生成の方法を説明します。

## 2.5.1 クロスエントロピーによる学習

Transformer による文章生成は、エンコーダの出力と直前までの出力を元に、次の単語を語彙の中から選択するという多クラス分類問題を繰り返し解いていると捉えることができます。そのため、Transformer の学習には多クラス分類でよく用いられるクロスエントロピー誤差関数を目的関数として用います。
クロスエントロピー誤差関数は正解となる確率分布と、モデル出力の確率分布を比較する関数で、[<図: 分布ごとのクロスエントロピーの違い>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示すように、分布の差が大きいほど値が大きくなります▲注▲。

- ▲注▲
    
    直感的には分布全体が少しシフトするだけであれば差は小さいですが、クロスエントロピーは必ずしも小さくなりません。クロスエントロピーは、各位置 (図中 -3, -2, …, 3 など) ごとの確率値の差を計算します。位置の近さを考慮した差を計算できる方法としては Wasserstein 距離などがあります。
    

![<図: 分布ごとのクロスエントロピーの違い>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/f4eb7656-1443-4efd-b0af-34930c41aa80/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_12x.png)

<図: 分布ごとのクロスエントロピーの違い>

文章生成タスクでは、正解データとして確率分布ではなく文章が与えられる場合が多いです。従って正解となる分布は、[<図: 単語予測時の正解分布>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示すような、1つの単語のみ確率が 1 で他が 0 になるような分布です。そのため、クロスエントロピー誤差は実質、モデルが正解単語の確率を高く出力するほど小さくなります。

![<図: 単語予測時の正解分布>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/739bbb6e-3eb1-454c-9c30-c0dbe66ae3cc/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_22x.png)

<図: 単語予測時の正解分布>

正解分布を $\bm p = (p_1, p_2, \ldots, p_N)$、モデルの出力分布を $\bm q = (q_1, q_2, \ldots, q_N)$ とすると、クロスエントロピー誤差 $H(\bm p, \bm q)$  は以下のようになります。

$$
H(\bm p, \bm q)=-\sum_{i=1}^N p_i\log q_i
$$

$-\log q_i$ は $q_i$ が 0 に近づくほど大きくなります。そのため、$p_i = 1$ の部分で $q_i$ の値が小さいと、誤差が大きくなります。

実装は [<コード: クロスエントロピーの実装>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のようにPyTorch に用意されているクロスエントロピーを計算するクラスを用います。実行すると、予測1 の方が予測2よりもクロスエントロピーが小さいことがわかります。

```python
import torch
from torch import nn

p1 = torch.tensor([1.0, 0.0, 0.0]) # 正解
p2 = torch.tensor([0.7, 0.2, 0.1]) # 予測1
p3 = torch.tensor([0.1, 0.2, 0.7]) # 予測2

cross_entropy = nn.CrossEntropyLoss()
print(cross_entropy(p1, p2)) # 0.8514
print(cross_entropy(p1, p3)) # 1.4514
```

## 2.5.2 Padding マスク と Subsequent マスク

2.4 節のエンコーダやデコーダの実装時、アテンション機構にいくつかのマスクが現れました。ここでは、これらのマスクの役割について説明します。

1 つ目はPadding マスクです。このマスクはバッチ学習を行うために必要になるマスクです。バッチ学習では、複数の入力をまとめて与えてることで計算を効率化します。その際に、与える入力は同じサイズになっている必要があります。つまり、文章の長さが $N_1, N_2, \ldots, N_k$ のようにバラバラの入力ではなく、すべて長さ $N$ に揃っていなければなりません。そこで用いるのがパディングです。[<図: パディング>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示すようにパディングでは、1つのバッチ内の最長の文に合わせて、**<pad>** トークンと呼ばれる特殊なトークンを追加します ▲注▲。

- ▲注▲
    
    図では省略していますが、実際は <bos> や <eos> を加えた後に <pad> を追加します。
    

![<図: パディング> ](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/a48ddfb0-88a2-473d-b382-bac15c9d16d0/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_42x.png)

<図: パディング> 

これによって例えば、バッチサイズが 32 で最長の文が 10 トークンであれば、入力を $32\times10$ の行列として扱えます。

[<図: パディング>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) の例では最長の文が 9 トークンであるため、1文目に 3 つの <pad> トークンを追加しています。しかし、最長の文が 10 トークンであれば、1文目には 4 つの <pad> トークンが追加されます。このように <pad> トークンは、たまたま同じバッチになった他の文に依存しており、<pad> トークン自体に文章を理解する上での意味はありません。 そのため、Transformer で処理する際は、このトークンを無視する必要があります。Padding マスクはこの、<pad> トークンを無視するためのマスクです。

アテンション機構で <pad> トークンを無視するために、内積アテンションの softmax 関数を利用します。softmax 関数は入力が $a_1^\prime, a_2^\prime, \ldots, a_n^\prime$ のとき、 $i$ 番目の出力は次のように与えられました。

$$
a_i = \frac{\exp(a_i^\prime)}{\sum_{j=1}^n \exp(a_k^\prime)}
$$

例として Padding マスクで $i$ 番目を無視したい場合、$a_i^\prime$ に非常に小さい値 ($-\infty$) を加えます。指数関数は中身が非常に小さい時、値が 0 に近づくため、アテンション重みが 0 として出力されます ([<図: パディングマスク>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21))。

![<図: パディングマスク>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/d58617b8-873c-48d0-bba9-12c749d1dd23/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_52x.png)

<図: パディングマスク>

パディングマスクは Transformer 中の全てのアテンションに必要になります。一方で Subsequent マスクは、デコーダにのみ適用されます。このマスクが必要になるのは学習時にデコーダに対して正解の文全体を入力するためです。つまり、学習時には[<図: 学習時の予測方法>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21)の下に示したような入力を与え、それらを1トークン分左にずらした出力を正解とします▲注▲。

- ▲注▲
    
    これによって学習時は、文章の長さ文繰り返す処理を省き、効率的に学習を進められます。
    
    このように、モデル自身の出力ではなく、正解のトークン列を過去の出力かのように入力することを Teacher Forcing といいます。
    

![<図: 学習時の予測方法>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/4e076644-4c95-4d28-8837-dfb23aab8c5a/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_92x.png)

<図: 学習時の予測方法>

図に示した通り、cat の予測には <bos> I am a までの入力しか参照してはいけません。これは、後ろの cat . まで参照できてしまうと単に一個後ろのトークンを持ってくることしか学習できなくなるためです。このような未来 (後続) のトークンを参照しないためのマスクが Subsequent マスクです。

Subsequent マスクもパディングマスクと同様に、無視したい位置に対して、softmax 関数への入力に $-\infty$ を加算します。具体的には、<図: Subsequent マスクの位置> に示したような上三角形部分が $-\infty$ になります▲注▲。

- ▲注▲
    
    マスクが行列になるのは、2.2.2 節の <図: 複数クエリ・複数キーに対する注意度の計算> で説明した通り、複数クエリを同時に扱うためです。
    

![<図: Subsequent マスクの位置>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/42253d4d-c241-4fcf-bd6f-4edd3509dcb2/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_102x.png)

<図: Subsequent マスクの位置>

アテンションには以上のように、大きく分けて2種類のマスクを用います。しかし、どちらのマスクもマスクによって無視する仕組み自体は同じため、アテンションの実装 ([<コード: マスクを考慮した内積アテンションの実装>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21))では1つのマスクとして受け取ります。

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        d_k = query.size(-1)
        # query の次元 (= キーの次元) でスケーリング
        # score.shape == (batch_size, query_len, key_len)
        score = torch.bmm(query, key.transpose(1, 2)) / (d_k**0.5)

        # マスクがある場合は, -infを代入してsoftmaxの値が0になるようにする
        # マスクは bool で受け取り、True の部分を -inf にする
        if mask is not None:
            score = score.masked_fill(mask, float("-inf"))

        weight = torch.softmax(score, dim=-1)
        output = torch.bmm(weight, value)

        return output
```

Transformer に関しては 2.4 節ですでにマスク付きの実装を示しているため、ここでは実装は省略します。2.4 節の実装における、各マスクが Transformer 内のどの部分で用いられるかを [<図: マスクの作用する部分>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示します。Subsequent マスクがデコーダ側のセルフアテンションにのみ付いていることに注意してください。これは、ソース・ターゲットアテンションではバリューとしてエンコーダ側の出力を使っており、未来の情報を含まないためです。

![<図: マスクの作用する部分>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/df4a2e88-e5f5-4011-97f8-20ccc26b1c9d/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_112x.png)

<図: マスクの作用する部分>

次に、各マスクの実装方法を説明します。パディングマスクは [<コード: パディングマスクの実装>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21)  に示すように、<pad> トークンの ID を受け取り、それと一致する部分を True にします。

```python
def create_padding_mask(pad_id: int, batch_tokens: Tensor):
		# batch_tokens.shape == (batch_size, sequence_length)
    mask = batch_tokens == pad_id
    mask = mask.unsqueeze(1)
    return mask
```

ここで batch_tokens はパディング済みのトークン列で、 `(バッチサイズ, 文章の長さ)` というサイズです。そのため、一致する部分を True としたマスクも同じサイズの Tensor になります。マスクは、バッチ内の各要素に対して行列である必要があるため、`unsqueeze` によって、 `(バッチサイズ, **1**, 文章の長さ)` の形に変更しています▲注▲。

- ▲注▲
    
    実際は`(バッチサイズ, 文章の長さ, 文章の長さ)` という形のマスクが必要がです。PyTorch の Tensor も NumPy 配列と同様にブロードキャストされるため、今回は2次元目が 1 の状態にしています。
    

Subsequent マスクの実装は [<コード: Subsequent マスクの実装>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のようになります。

```python
def create_subsequent_mask(batch_tokens: Tensor):
    sequence_len = batch_tokens.size(1)
    mask = torch.triu(
        torch.full((sequence_len, sequence_len), 1),
        diagonal=1,
    )
    mask = mask == 1
    mask = mask.unsqueeze(0)
    return mask
```

`torch.triu` は受け取った行列に対して上三角形部分以外を 0 にする関数です。diagonal 引数は対角線から何個の幅を追加で 0 にするかを指定します。今回の場合、対角線のみ追加で 0 にしたいので diagonal=1 としました。マスクとしては True/False の状態にしたいため、その後に値が 1 の部分を True にしています。最後の unsqueeze はマスクを `(1, 文章の長さ, 文章の長さ)` という形に変更しています。

## 2.5.3 Transformer の学習

これまでの節で、学習時に必要な内容は全て揃いました。ここでは、実際のデータを用いて Transformer を学習する方法を説明します。

データセットとしては、田中コーパスという日英の対応する文章コーパスを前処理した small_parallel_enja を利用します。このデータでは、文章のトークン化や小文字化などがされています。田中コーパスに含まれる文章の例を <図: 田中コーパスのデータのサンプル> に示します。

![スクリーンショット 2024-05-30 7.13.40.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/bb1c8f3a-31ef-49a1-bf2d-0b31102b6124/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88_2024-05-30_7.13.40.png)

![スクリーンショット 2024-05-30 7.13.45.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/89030a85-b3b1-408a-bb12-f65514bf945f/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88_2024-05-30_7.13.45.png)

<図: 田中コーパスのデータのサンプル>

small_parallel_enja のデータは1万件ずつに分割されています。まずは動作確認のため、最初の1万件のみを用いて学習を行うため [<コード: データセットのダウンロードとパスの設定>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のように設定してください。

```python
data_dir = Path("small_parallel_enja")
if not data_dir.exists():
    !git clone https://github.com/odashi/small_parallel_enja.git {data_dir}

train_ja = data_dir / "train.ja.000"
train_en = data_dir / "train.en.000"
```

small_parallel_enja では、1文が1行になっており、トークンはスペース区切りになっています。そのため、学習データの読み込みは [<コード: 学習データの読み込み>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のようにできます。ただし、<bos>、<eos> トークンは付いていないため、読み込み時に先頭と末尾に付与する必要があります。

```python
def iter_corpus(
    path: Path,
    bos: str | None = "<bos>",
    eos: str | None = "<eos>",
) -> Iterator[list[str]]:
    with path.open("r") as f:
        for line in f:
            if bos:
                line = bos + " " + line
            if eos:
                line = line + " " + eos
            yield line.split()

train_tokens_ja = [tokens for tokens in iter_corpus(train_ja)]
train_tokens_en = [tokens for tokens in iter_corpus(train_en)]
```

ここで、train_tokens_ja、train_tokens_en はそれぞれ文章ごとに単語を分割したリストのリストになっています。これらのリストの要素の1つを抜き出すと、以下のようなリストになっています。

['<bos>', '誰', 'が', '一番', 'に', '着', 'く', 'か', '私', 'に', 'は', '分か', 'り', 'ま', 'せ', 'ん', '。', '<eos>']

['<bos>', 'i', 'can', "'", 't', 'tell', 'who', 'will', 'arrive', 'first', '.', '<eos>']

では、分割した単語リストからボキャブラリーを作成しましょう。ボキャブラリーとして重複なく単語のリストを持つことで、各単語に一意な数字 (ID) を割り振ることができます。これには、torchtext の build_vocab_from_iterator 関数を利用します。[<コード: ボキャブラリーの作成>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示すようにこの関数には specials として特殊トークンを渡します。特殊トークンとは、すでに現れた <bos>、<eos>、<pad> などの、単語としてではなく、学習や推論時に何らかの機能を持つトークンです。これらのトークンに加えて、新たに現れた <unk> トークンがあります。このトークンは未知の単語、つまりボキャブラリーに無い単語を割り当てるトークンです。そのため、ボキャブラリーを作成した後に、デフォルトとして <unk> トークンのIDを割り振るように設定しています。

```python
from torchtext.vocab import build_vocab_from_iterator

vocab_ja = build_vocab_from_iterator(
    iterator=train_tokens_ja,
    specials=("<unk>", "<pad>", "<bos>", "<eos>"),
)
vocab_ja.set_default_index(vocab_ja["<unk>"])
vocab_en = build_vocab_from_iterator(
    iterator=train_tokens_en,
    specials=("<unk>", "<pad>", "<bos>", "<eos>"),
)
vocab_en.set_default_index(vocab_en["<unk>"])
```

では、[<コード: 単語変換辞書>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) を実行して作成したボキャブラリーで文章を ID 列に変換してみましょう。

```python
print("<unk>:", vocab_ja["<unk>"]) # <unk>: 0
tokens = ["<bos>", "吾輩", "は", "猫", "で", "ある", "<eos>"]
for token in tokens:
    print(vocab_ja[token], end=" ")
```

実行すると2つ目の print では `2 0 5 437 12 667 3`  のように出力されます。1つ目の print で表示した通り、<unk> トークンの ID は 0 になるので、今回のデータには 吾輩 という単語が入っておらず、未知語として扱われていることがわかります。

次に、PyTorch を用いてデータをロードするためのクラスを作成します。データをロードする際は、バッチ学習をするために、先述の通り <pad> トークンを追加します。<pad> トークンをいくつ追加するかはロードされたバッチによって異なります。データを読み込むたびに前処理を行う方法として、PyTorch の DataLoader クラスには collate_fn という引数が用意されています。まずはこれを実装しましょう ([<コード: collate_fn の実装>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21))。

```python
from torchtext import transforms

src_transforms = transforms.Sequential(
    transforms.VocabTransform(vocab_ja),
    transforms.ToTensor(padding_value=vocab_ja["<pad>"]),
)
tgt_transforms = transforms.Sequential(
    transforms.VocabTransform(vocab_en),
    transforms.ToTensor(padding_value=vocab_en["<pad>"]),
)

def collate_fn(batch: Tensor) -> tuple[Tensor, Tensor]:
    src_texts, tgt_texts = [], []
    for s, t in batch:
        src_texts.append(s)
        tgt_texts.append(t)

    src_texts = src_transforms(src_texts)
    tgt_texts = tgt_transforms(tgt_texts)

    return src_texts, tgt_texts
```

collate_fn の引数 batch には、`[(日本語文1, 英語文1), (日本語文2, 英語文2), …]` といった形でデータが格納されています。最初の for 文ではこれを日本語文のリストと英語分のリストに分解しています。その後、コード上部で定義している src_transforms や tgt_transforms という変換を適用しています。変換の内容としては、単語から ID への変換、バッチに含まれる最長のテキストに合わせた <pad> トークンの追加と PyTorch Tensor への変換の 2 つを行なっています。

上述の collate_fn を利用して、DataLoader クラスを作成するコードを [<コード: DataLoader の作成>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示します。

```python

from torch.utils.data import DataLoader

train_loader = DataLoader(
    list(zip(train_tokens_ja, train_tokens_en)), # 文章データ
    batch_size=64, # バッチサイズ
    shuffle=True, # データをシャッフルする
    collate_fn=collate_fn, # パディングの追加、ID の Tensor に変換
)
```

各引数の内容はコメントに書いている通りです。[<コード: DataLoader の動作確認>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) を実行して想定通りにデータを取得できるかを検証してみましょう。

```python
batch = next(iter(train_loader))
src_texts, tgt_texts = batch
print(src_texts.shape) # torch.Size([16, 18])
print(tgt_texts.shape) # torch.Size([16, 15])
```

shape の第1要素はバッチサイズなので、16 と出力されるはずです。第2要素はそのバッチ内で最も長い文の長さに揃えられるので、実行のたびに異なります。

では、いよいよ Transformer のインスタンスを作成します。今回の実装 ([<コード: Transformer インスタンスの作成>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21)) ではトークン埋め込みは 512 次元、ブロックを繰り返す回数は 6 回、マルチヘッドアテンションのヘッド数は 8 に設定します。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "を使用")

embedding_dim = 512
n_blocks = 6
n_heads = 8
expansion_rate = 1

# 語彙数を取得
src_vocab_size = len(vocab_ja)
tgt_vocab_size = len(vocab_en)

# 最も長い文章の長さを取得
max_len_ja = len(max(train_tokens_ja, key=lambda x: len(x)))
max_len_en = len(max(train_tokens_en, key=lambda x: len(x)))
max_length = max(max_len_ja, max_len_en)

model = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    max_sequence_len=max_length,
    d_model=embedding_dim,
    n_blocks=n_blocks,
    n_heads=n_heads,
    d_k=embedding_dim,
    d_v=embedding_dim,
    d_ff=embedding_dim * expansion_rate,
).to(device)
```

損失関数や最適化手法、スケジューラ等の学習の設定は [<コード: 学習の設定>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のようにします▲注▲。ここではクロスエントロピー誤差に ignore_index という引数を設定しています。これにより、<pad> トークンの部分では学習を行わなくなります。

- ▲注▲
    
    これらの設定について不明な場合は付録の PyTorch 入門を参照してください。
    

```python
PAD_ID = vocab_ja["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # クロスエントロピー誤差
lr = 0.0001  # 学習率
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.95)
```

学習を行う関数は [<コード: モデルを学習するための関数>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のように定義します。

```python
def train(model: nn.Module, log_interval: int = 10) -> list[float]:
    model.train()
    loss_history = []
    for i, (src_texts, tgt_texts) in enumerate(train_loader):
        # tgt の入力は最後の単語を除く
        tgt_input = tgt_texts[:, :-1]
        # tgt の出力は最初の単語を除く
        tgt_output = tgt_texts[:, 1:]
        src_padding_mask = create_padding_mask(PAD_ID, src_texts)
        tgt_padding_mask = create_padding_mask(PAD_ID, tgt_input)
        tgt_subsequent_mask = create_subsequent_mask(tgt_input)
        tgt_mask = tgt_padding_mask + tgt_subsequent_mask
        # Tensor のデバイスを設定
        src_texts, tgt_input, tgt_output = (
            src_texts.to(device),
            tgt_input.to(device),
            tgt_output.to(device),
        )
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        # モデル出力を取得
        out = model(src_texts, tgt_input, src_padding_mask, tgt_mask, src_padding_mask)
        # 出力と教師データを1次元に変換
        out_flat = out.view(-1, out.size(-1))
        tgt_flat = tgt_output.flatten()
        # 誤差関数を計算
        loss = criterion(out_flat, tgt_flat)
        optimizer.zero_grad()
        # 誤差逆伝播
        loss.backward()
        optimizer.step()
        if (i + 1) % log_interval == 0:
            print(f"step {i+1}: train loss = {loss.item()}")
        loss_history.append(loss.item())
    return loss_history
```

ループの先頭での tgt_input、tgt_output の取得方法は <図: 学習時の予測方法> で説明した通りです。マスクは [<図: マスクの作用する部分>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) にある通りのものを用います。ただし src_tgt_padding_mask は src_padding_mask と同じため、使いまわしています。

出力次元の変換では `(バッチサイズ, 文章の長さ, 語彙数)` という3次元から `(バッチサイズ×文章の長さ, 語彙数)` という2次元に変換しています。一方、教師データの方は `バッチサイズ×文章の長さ` というサイズの1次元に変更しています。2.5.1 節で説明した通り、出力側は語彙数分の要素を持つ確率分布で、教師側は1つの ID のみ確率1の確率分布なので、このようなサイズになっています。また、上述のような変換が必要なのは、PyTorch の誤差関数が `(バッチサイズ, 出力要素数)` という形の入力を受け取るためです。

いよいよ Transformer を学習します。今回は全データに対する学習を合計20エポック回します。すでに学習用の関数は定義しているので実装は [<コード: 学習>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のようにシンプルになります。

```python
from tqdm.auto import tqdm

n_epochs = 20  # エポック数
pbar = tqdm(total=n_epochs)
for epoch in range(n_epochs):
    pbar.update(1)
    pbar.set_description(desc="Epoch")
    train(model)
```

これまでのコードが適切に実行できていると、10ステップごとに表示される誤差関数の値が下がっていくはずです。学習の全体での誤差推移の一例を [<図: 学習ステップと誤差関数の値>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示します。色の切り替わる点はエポックが切り替わっている部分です。

![<図: 学習ステップと誤差関数の値>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/b2fa8255-244b-4061-8e49-61180ac7e91d/loss_history_(1).png)

<図: 学習ステップと誤差関数の値>

学習が完了したら、新たな文章の翻訳をしてみましょう。推論には 2.4 節の最後に実装した inference メソッドを利用して [<コード: 推論>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のように実装できます。出力はトークン ID になるので、はトークン ID から単語へ変換するために `itos` というオブジェクトを使用します。

```python
# 入力の準備
text = "<bos> 今日 の 天気 は 晴れ です 。 <eos>"
tokens =  text.split()
input_tokens = src_transforms([tokens]).to(device)
# 推論
tgt_tokens = model.inference(
    input_tokens, bos_token=vocab_ja["<bos>"], eos_token=vocab_ja["<eos>"]
)
# 出力の取得
itos = vocab_en.get_itos()
text = " ".join(itos[token_id] for token_id in tgt_tokens[0])
print(text) # <bos> it is fine today . <eos>
```

サンプルコードでは `今日の天気は晴れです。` のように簡単なテキストを入れて、 `it is fine today` と正しく翻訳できました ▲注▲。他にも様々なテキストを入力してみて、今回学習した Transformer がどの程度の翻訳能力を獲得したか試してみてください。

- ▲注▲
    
    学習時の初期値や与える順番のランダム性から、具体的な翻訳結果は変わる可能性もあります。
    

## 2.5.4 発展的な推論方法

2.4 節で説明した推論方法では、トークンレベルで最も確率の高いものを選択していました。このように、1ステップごとに最大のものを選ぶ方法を貪欲法といいます。ここでは、貪欲法以外の生成方法として以下の3つを順番に紹介します。

- ビームサーチ
- top-k
- top-p

[<図: 貪欲法が最適でない場合>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) は、文章生成時の各ステップで選ぶ単語に関する分岐を示しています▲注▲。図の例では貪欲法だと、 `it` の確率が最も高いので `it` が選択されます。一方で、4単語目までをみると `<bos> the weather will` の方が全体としての確率は `it` から始めた時の確率よりも高くなります。貪欲法ではこのような、短期的には確率が低いものの、長期的には良い文章を見つけることができません。

- ▲注▲
    
    実際は全語彙に関する分岐があります。
    

![<図: 貪欲法が最適でない場合>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/96712d8c-6458-49f2-b4d0-d0ee2c6a0220/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_182x.png)

<図: 貪欲法が最適でない場合>

長期的に良い文章を探すためには、あらゆる文章の確率を計算して確率が最大になるものを選ぶという方法が考えられます。しかし、これを行うと調べなければならないパターンが膨大になってしまいます。例えば、語彙数が 5,000 だとしても、 [<図: 組み合わせ爆発>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示す通り、2ステップ調べるだけで組み合わせは 25,000,000 通りです。10ステップにもなると約 $10^{37}$ 通りになってしまいます。

![<図: 組み合わせ爆発>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/218e9c8c-2e70-403a-b307-cbf3fed4856e/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_262x.png)

<図: 組み合わせ爆発>

そこで、ビームサーチでは [<図: ビームサーチ>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のように各ステップで確率の高いものに絞りその次のステップを探索します。このとき、探索するルートとして残す数をビーム幅といいます。図の例ではビーム幅は2としています。

![<図: ビームサーチ>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/81a5bd68-261b-4efc-aac5-177ae496dff4/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_272x.png)

<図: ビームサーチ>

ビームサーチによる文章生成の実装を [<コード: ビームサーチ>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示します。

```python
from torch.nn import functional as F

def beam_search_inference(
    model: nn.Module,
    src: Tensor,
    bos_token: int,
    eos_token: int,
    beam_width: int = 5,
    max_length: int = 50,
) -> Tensor:
    device = src.device
    encoder_output = model.encoder(src)

    # 初期状態の作成
    sequences = [[bos_token]]
    scores = torch.zeros(1, device=device)
    ended_seq_mask = [False]

    for _ in range(max_length):
        all_candidates = []
        for i in range(len(sequences)):
            seq = sequences[i]
            if ended_seq_mask[i]:
                # 既に終了しているシーケンスはそのまま保持
                all_candidates.append((scores[i], seq))
                continue

            tgt = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            decoder_output = model.decoder(tgt, encoder_output)
            logits = model.linear(decoder_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # 現在のスコアと次のトークンの確率を足して全候補を生成
            for j in range(log_probs.size(1)):
                candidate = seq + [j]
                candidate_score = scores[i] + log_probs[0, j]
                all_candidates.append((candidate_score, candidate))

        # ビーム幅でソートしてトップ beam_width 個を選択
        top_candidates = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)[
            :beam_width
        ]
        sequences = [x[1] for x in top_candidates]
        scores = torch.tensor([x[0] for x in top_candidates], device=device)
        ended_seq_mask = [seq[-1] == eos_token for seq in sequences]

        # 全ての候補が終了トークンで終わっている場合、終了
        if all(ended_seq_mask):
            break

    # スコアが最も高い候補を選択
    best_sequence = sequences[0]
    return torch.tensor(best_sequence, device=device)
```

このコードではまず、 `<bos>` トークンのみを持つ初期状態を作成します。その後、探索中の全ての単語列 (シーケンス) が `<eos>` に達するか、元から設定していた最大の長さに達するまでループを回します。このループ内では、新たな単語の確率を計算し、その確率の対数を取って前の単語までの対数尤度と足し合わせます ▲注▲。ただし、すでに `<eos>` に達しているシーケンスについては次の単語は計算しません。このようなシーケンスと、新たな単語を加えたシーケンスのリストから、対数尤度が大きい順にビーム幅分を残して次のステップに移行します。

- ▲注▲
    
    ある単語列が現れる尤度は、その列の単語全てに対してモデルの出力する確率の積 $p_1\cdot p_2\cdot p_n$ です。対数関数は単調増加なので、対数をとっても大小関係は変わりません。
    

では、 `beam_search_inference` 関数を用いて文章を生成してみましょう。

```python
text = "<bos> 今日 の 天気 は 晴れ です 。 <eos>"
tokens = text.split()
input_tokens = src_transforms([tokens]).to(device)
tgt_tokens = beam_search_inference(
    model, input_tokens, vocab_en["<bos>"], vocab_en["<eos>"], max_length=20
)
itos = vocab_en.get_itos()
text = " ".join(itos[token_id] for token_id in tgt_tokens)
print(text)  # <bos> the weather is fine today . <eos>
```

貪欲法に比べると生成に時間がかかったと思います。貪欲法では単に語彙数×ステップ数の探索でしたが、ビームサーチではこれがビーム幅倍されるためです。出力された文章については、例文のレベルでは大差なかったかもしれません。貪欲法と同様に、さまざまなテキストを入力してみて、実際にどのような傾向があるのか試してみてください。

貪欲法もビームサーチによる生成も、何度実行しても同じ文章が生成されるような、決定的なアルゴリズムになっていました。一方、以降で紹介する top-k、top-p はモデルの出力する確率を用いたサンプリングを行う、非決定的なアルゴリズムです。人間の書く文章は、尤度を最大化する決定的なアルゴリズムに比べて、単語ごとにみると確率の低いものも用いていることが知られています。そのため、物語などのクリエイティブな文章を生成させたいときは、単に尤度を最大化するのではなくサンプリングを用いたアルゴリズムの利用が有効です。

top-k や top-p の説明の前に、まず温度（temperature）パラメータを用いたサンプリングについて説明します。 ChatGPT 等を API 経由で利用したこのとある人はこのパラメータを見たことがあると思います。温度パラメータは Transformer の最後の線形層の出力に対する softmax に用いられるもので、温度 $T$ の softmax 関数は次式のようになります。

$$
\mathrm{softmax}(\bm x, T)=\left(\frac{\exp(x_1/T)}{\sum_{j=1}^n \exp(x_j/T)}, \ldots\right)
$$

温度パラメータによる、分布の変化を [<図: 温度パラメータごとの分布の変化>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示します。上式からもわかるように $T=1$ の場合が元の分布です。図はランダムに作成した分布に対して、$T=0.1, 1.0, 2.0$ の場合をプロットしたものです。

![<図: 温度パラメータごとの分布の変化>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/fe40bbb6-7561-49b2-b17a-54baebd77cc3/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_372x.png)

<図: 温度パラメータごとの分布の変化>

図からわかるように、温度が高くなるほど確率の差がなくなり、温度が低いほど差が大きくなっています。そのため、ランダム性を高くしたければ温度を高くし、決定的なサンプリングに近づけたければ温度を低く設定します。特に $T=0$ の場合は、サンプリングを行わず単なる貪欲法による文章生成を行うのと同じになります。

温度パラメータ付きのサンプリングによる文章生成を実装してみましょう ([<コード: 温度パラメータ付きのサンプリング>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21))。

```python
@torch.inference_mode
def temperature_inference(
    model: nn.Module,
    src: Tensor,
    bos_token: int,
    eos_token: int,
    temperature: float = 1.0,
    max_length: int = 50,
):
    tgt_tokens = torch.tensor([[bos_token]]).to(src.device)

    encoder_output = model.encoder(src)
    for _ in range(max_length):
        decoder_output = model.decoder(tgt_tokens, encoder_output)
        score = model.linear(decoder_output)
        # 温度パラメータによる変換
        score = score / temperature
        porbability = F.softmax(score[0, -1], dim=-1)
        # トークンをサンプリング
        pred = torch.multinomial(porbability, 1)
        tgt_tokens = torch.cat((tgt_tokens, pred), axis=-1)
        if pred[0, 0].item() == eos_token:
            break

    return tgt_tokens
```

文章の生成部分はこれまでと同様ですが、今回はサンプリングを用いているため、毎回異なる出力が得られます。複数回試して、どのような文章が生成されるか確認してください。

上述のサンプリングでは、モデルの出力した全単語に対する確率分布から、次の単語をサンプリングしていました。これに対して、top-k サンプリングではまず、モデルの出力した確率分布のうち、上位 k 個のみを取り出します。その後、取り出した k 個に関して正規化した分布からサンプリングを行います。

例えば語彙数が $N$ として、各単語の確率が大きい順に $(p_1, p_2, \ldots, p_N)$ の場合、以下のように確率分布を変換します。

$$
\hat p_i = \begin{cases}
\frac{p_i}{\sum_{j=1}^N p_j} & i\leq k\\
0 & i > k
\end{cases}
$$

`<bos> it is` の次の単語と `<bos> it is fine` の次の単語の出力確率を[<図: 元の出力確率>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21)に示します。top-k で k=3 として変換された確率分布は [<図: top-k の出力確率>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) のようになります。

![<図: 元の出力確率>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/61493a2a-2df3-48d8-a0c9-31b95c434643/output_probability.png)

<図: 元の出力確率>

![<図: top-k の出力確率>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/7671d030-3281-4307-8c90-29e21a15b27c/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_392x.png)

<図: top-k の出力確率>

[<コード: top-k サンプリング>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示すように実装では、確率を大きい順に変換する際にインデックスの並べ替えを先に計算します▲注▲。

- ▲注▲
    
    確率分布だけソートしてしまうとサンプリング時に ID が 0 ~ k のトークンからサンプリングしてしまうためです。
    

```python
@torch.inference_mode
def top_k_inference(
    model: nn.Module,
    src: Tensor,
    bos_token: int,
    eos_token: int,
    temperature: float = 1.0,
    k: float = 5,
    max_length: int = 50,
):
    tgt_tokens = torch.tensor([[bos_token]]).to(src.device)

    encoder_output = model.encoder(src)
    for _ in range(max_length):
        decoder_output = model.decoder(tgt_tokens, encoder_output)
        score = model.linear(decoder_output)
        # 温度パラメータによる変換
        score = score / temperature
        probability = F.softmax(score[0, -1], dim=-1)

        # 確率の高い順にソートしたインデックスを取得
        idx_sorted = torch.argsort(probability, descending=True)
        # 上位k個のインデックスを取得
        idx_k = idx_sorted[:k]
        # 上位k個の確率を取得
        p_k = probability[idx_k]
        # 正規化
        p_k /= torch.sum(p_k)
        # トークンをサンプリング
        pred = torch.multinomial(p_k, 1)
        pred = idx_k[pred].unsqueeze(0)
        tgt_tokens = torch.cat((tgt_tokens, pred), axis=-1)
        if pred[0, 0].item() == eos_token:
            break

    return tgt_tokens
```

このコードを用いて、複数回サンプリングしてみてください。単に温度パラメータを利用するサンプリングに比べると、適切な文章を生成しやすいはずです。ただし、今回は学習データ数を抑えているため、top-k でも文章として成立していないサンプルも出力されるかもしれません。

top-k は、取り出す単語の個数 k を指定していました。これに対して top-p では、個数は指定せずに上位から確率の合計が p 以上になるまでの単語を取り出します。<図: 確率に極端な偏りがある場合> に top-k の問題点と top-p のアイディアを示します。

図のように、top-k では非常に低い確率の単語 (you) も選ばれてしまう可能性があります。一方、top-p では確率が高い部分 (clear) で閾値を超えるため、確率の低い単語は選択されません。

![<図: 確率に極端な偏りがある場合>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/b87cf9da-334d-4dec-afcf-dd7231c77649/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_382x.png)

<図: 確率に極端な偏りがある場合>

top-p によるサンプリングの実装を [<コード: top-p サンプリング>](https://www.notion.so/2-5-Transformer-b2f9172e681545d0aa962ec5c0fbe054?pvs=21) に示します。

```python
@torch.inference_mode
def top_p_inference(
    model: nn.Module,
    src: Tensor,
    bos_token: int,
    eos_token: int,
    temperature: float = 1.0,
    p: float = 0.8,
    max_length: int = 50,
):
    tgt_tokens = torch.tensor([[bos_token]]).to(src.device)

    encoder_output = model.encoder(src)
    for _ in range(max_length):
        decoder_output = model.decoder(tgt_tokens, encoder_output)
        score = model.linear(decoder_output)
        # 温度パラメータによる変換
        score = score / temperature
        porbability = F.softmax(score[0, -1], dim=-1)
        idx_sorted = torch.argsort(porbability, descending=True)
        p_sorted = porbability[idx_sorted]
        # ソートされた確率の累積和を計算
        p_cumsum = torch.cumsum(p_sorted, dim=-1)
        # p を超える最初のインデックスを取得
        idx = torch.sum(p_cumsum < p).item() + 1
        # インデックスが範囲内に収まるように調整
        idx = min(idx, len(p_cumsum) - 1)
        # p を超えない範囲で上位の確率分布を取得
        p_top = p_sorted[:idx]
        # 正規化
        p_top = p_top / torch.sum(p_top)
        # トークンをサンプリング
        pred = torch.multinomial(p_top, 1)
        pred = idx_sorted[pred]
        tgt_tokens = torch.cat((tgt_tokens, pred), axis=-1)
        if pred[0, 0].item() == eos_token:
            break

    return tgt_tokens
```

top-p によるサンプリングについても、複数回実行して、どのような出力が得られるか確認してみましょう。top-k と比べるとおかしな文章を生成することは少ないはずです。
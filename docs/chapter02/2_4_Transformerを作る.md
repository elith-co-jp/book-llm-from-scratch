# 2.4 Transformer を作る

ここまで Transformer を構成する主要な要素について説明してきました。本節ではそれらの要素を組み合わせて Transformer モデル全体を実装していきます。

## 2.4.1 エンコーダ

エンコーダの役割は、入力された文章を特徴量のベクトルに変換することです。具体的には、文章中の各単語を埋め込みベクトルに変換し、各単語の周辺単語との関係性を考慮しながら、文章全体の意味を表現するベクトルを生成します。

エンコーダは、以下の要素で構成されます。

1. **トークン埋め込み**: 単語をベクトルに変換
2. **位置エンコーディング**: 単語の位置情報を付加
3. **Self-Attention**: 各単語と他の単語との関係性を計算
4. **Feed Forward Network**: Self-Attentionの出力を非線形変換

ここで新たに現れた Self-Attention は、クエリとなる入力と、キー・バリューになる入力が同一である Multi-Head Attention のことです。入力となる文章の各ワードを解釈するために~~が~~、その文章内で注目するべき部分を計算するため Self-Attention と呼ばれます。これの対となる概念の Source-Target Attention については 2.4.2 節で扱います。

![<図: Self-Attention>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/8bea49cf-5d45-4a7a-995d-1684e522e4f1/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_42x.png)

<図: Self-Attention>

![<図: Self-Attentionの概念図>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/f03b81b4-b5cc-4324-bf10-d4949f5c0a53/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_62x.png)

<図: Self-Attentionの概念図>

> <図: Self-Attention>
右側の図を用いる場合、各単語は位置を表しているのであって、その単語自体を入力するわけではない旨を脚注に追加。
> 

以上のように構成される エンコーダ の全体像を [<図: エンコーダ>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) に示します。

![<図: エンコーダ>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/ffc9bf6d-e272-4f2e-b1ff-f72f80ef4c73/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_12x.png)

<図: エンコーダ>

図中で $\times 6$ と書いているのは、点線で囲まれたブロックを 6 回繰り返す、つまり [<図: エンコーダを展開した図>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) のようになっていることを表します。

![<図: エンコーダを展開した図>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/78e4eb59-d281-4da3-aaa7-6b3528758022/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_22x.png)

<図: エンコーダを展開した図>

このようにアテンションによる他の位置の中間表現との関係性の計算と全結合層による各位置の表現の変換で構成された エンコーダブロックを複数積み重ねることで、入力文章の特徴量を抽出します。

エンコーダは入力されたベクトルと同じ次元のベクトルを入力文章の長さ分だけ出力します。つまり、トークン埋め込みベクトルが 512 次元で 10 トークンある場合、512 次元のベクトルを 10 個出力します。入力時には、位置ごとのトークンの情報とそのトークンの位置の情報しか持ちませんでしたが、出力時の 10 個のベクトルはそれぞれが文全体を考慮した特徴表現になっています。

では、エンコーダを実装してみましょう。まず必要なのは <図: エンコーダ> の点線で囲まれたエンコーダブロックです。[<コード: エンコーダブロック>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) にその実装を示します。

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:
        x_attention = self.attention(x, x, x, mask=src_padding_mask)
        x = self.layer_norm1(x + x_attention)
        x_ff = self.feed_forward(x)
        x = self.layer_norm2(x + x_ff)
        return x
```

エンコーダブロックは `d_model` 、 `n_heads` 、 `d_k` 、 `d_v` 、 `d_ff` という5つの引数をもとに作成されます。 `d_model` はエンコーダブロックの入出力となるベクトルの次元です。 `n_heads` はマルチヘッドアテンションで幾つのヘッドを用意するかを表しています。　`d_k` 、 `d_v` は 2.2.4 節の <図: マルチヘッドアテンションの全体像> における、$K^\prime$ や $V^\prime$ で表されるキー・クエリを線形変換した後の次元数です。これらは通常、 `d_model / n_heads` に設定されます。クエリの次元を設定していないのは、内積アテンションにおいてクエリとキーの次元は同じである必要があるためです。最後の引数 `d_ff` は FNN の中間ユニットの次元数です。これを大きくすることで、FNN に `d_model` によらない表現力を与えることができます。各メソッドの中身については、<図: エンコーダ> のエンコーダブロックの繋がり方をそのまま実装したものになっていますので、図と照らし合わせて確認してみてください。

forward メソッドではマスクを受け取って、それをセルフアテンションに渡しています。2.2 節で実装したアテンションではこのようなマスクはありませんでした。これについては 2.5 節で新たに解説と実装をします。以降でもいくつかのクラスでマスクを利用しますが、本節では利用しないため、無視してください。

エンコーダブロックの実装が完了したので、次はエンコーダそのものを作ります。エンコーダの実装は [<コード: エンコーダ>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) のとおりです。

```python
class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.pe = PositionalEncoding(d_model, max_sequence_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:
        x = self.embedding(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, src_padding_mask=src_padding_mask)
        return x
```

コンストラクタの引数には、エンコーダブロックに与えるものの他に、語彙数 ( `vocabulary_size` )、入力文章の長さの上限 ( `max_sequence_len` ) 、エンコーダブロックを繰り返す回数 (`n_blocks` ) を与えます。

## 2.4.2 デコーダ

デコーダの役割は、エンコーダによって生成された特徴量ベクトルから、出力単語列 (文章) を生成することです。Decoder によるテキストの生成は自己回帰的 (Autoregressive) な生成と呼ばれ、エンコーダ出力に自身の過去の出力を加えて新たな出力を得るというプロセスを繰り返し行うことで文章を生成します。[<図: Decoder による出力の生成>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) にこのような文章生成の過程を示します▲注▲。

- ▲注▲
    
    図中で、新しい単語以外も出力されていることに注意してください。アテンションの構造上、エンコーダもデコーダも入力された系列と同じ長さの出力が得られます。
    

![<図: Decoder による出力の生成> ](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/ade12cc1-d0d8-466c-8c33-3904890faef6/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_82x.png)

<図: Decoder による出力の生成> 

図にあるように、デコーダに最初、文章の始まり (Begin of Sentence) を表す **<bos>** というトークンが入力されます。これに対して、デコーダは最初の単語 I を出力します。次のステップでは、直前の出力である I を加えて <bos> I を入力し、I am という出力を得ます。このような計算を繰り返して、デコーダが文章の終わり (End of Sentence) を表す **<eos>** を出力したら生成を終了します。

エンコーダは、文の全体を入力して一括の処理で出力を計算することができました。これに対してデコーダは上述のように、1語ずつ出力します。ChatGPT 等のウェブアプリ等で、出力が前から順に少しずつ表示されるのは、このような計算で出力が得られるたびに表示しているためです。

デコーダは、以下の要素から構成されます。

1. **出力埋め込み**: 単語をベクトルに変換
2. **位置エンコーディング**: 単語の位置情報を付加
3. **Self-Attention**: 各単語と他の単語との関係性を計算
4. **Source-Target Attention**: エンコーダの出力と、デコーダの中間表現の関係性を計算
5. **Feed Forward Network**: 中間表現を非線形変換

出力埋め込み、位置エンコーディングはエンコーダと同様です。エンコーダと大きく異なるのは Sorce-Target Attention という部分です。Self-Attention クエリ・キー・バリューに入力される文は同じでした。これに対して Source-Target Attention はソースとなるキー・バリュー、ターゲットであるクエリにそれぞれ別の文を用います。Transformer では、ソース側にエンコーダの出力、ターゲット側にデコーダの中間表現を入力します。

![<図: ソース・ターゲットアテンション>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/e4b6e8d3-ea08-419f-b04c-2d2dec12650e/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_52x.png)

<図: ソース・ターゲットアテンション>

![<図: ソース・ターゲットアテンションの概念図>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/a9b9940e-99d6-406d-b395-72a93b29a4ef/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_6_12x.png)

<図: ソース・ターゲットアテンションの概念図>

以上のように構成される デコーダ の全体像を [<図: デコーダ>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) に示します。また、エンコーダ同様、デコーダの ×6 の部分を展開した図を [<図: デコーダブロックを展開した図>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) に示します。

![ <図: デコーダ> ](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/76f3e978-0c0e-4c00-94f3-fbfdc55bde89/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_222x.png)

 <図: デコーダ> 

![<図: デコーダブロックを展開した図>](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/b06becc9-7e96-492d-b69b-0658255e5c92/%E3%82%A2%E3%83%BC%E3%83%88%E3%83%9B%E3%82%99%E3%83%BC%E3%83%88%E3%82%99_212x.png)

<図: デコーダブロックを展開した図>

[<図: デコーダブロックを展開した図>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) からわかるように、各デコーダブロックのソース・ターゲットアテンションに与えられるエンコーダ出力は同じものです。

デコーダブロックとデコーダ自体の実装を [<コード: デコーダブロック>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) 、[<コード: デコーダ>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) に示します。特に、デコーダブロックの `forward` メソッドで `self.attention_source_target` に入力されている内容から Source-Target Attention になっていることを確認してください。

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.attention_source_target = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm3 = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x_attention = self.attention(x, x, x, mask=tgt_mask)
        x = self.layer_norm1(x + x_attention)
        x_attention_source_target = self.attention_source_target(
            x, encoder_output, encoder_output, mask=src_tgt_padding_mask
        )
        x = self.layer_norm2(x + x_attention_source_target)
        x_ff = self.feed_forward(x)
        x = self.layer_norm3(x + x_ff)
        return x
```

```python
class Decoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.pe = PositionalEncoding(d_model, max_sequence_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                tgt_mask=tgt_mask,
                src_tgt_padding_mask=src_tgt_padding_mask,
            )
        return x

```

## 2.4.3 Transformer

Transformer の実装を [<コード: Transformer>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) に示します。これまでにエンコーダとデコーダを実装しているので、これらを組み合わせて構築します。また、[<図: Transformer>](https://www.notion.so/2-1-Transformer-2bcdf358370e46858ffd199a675e912b?pvs=21) にも示した通り、デコーダの出力後に全結合層があります。この全結合層の入力は埋め込みベクトルの次元数、出力は扱える語彙数です。学習や推論の都合上、実装には入れていませんが [<図: Transformer>](https://www.notion.so/2-1-Transformer-2bcdf358370e46858ffd199a675e912b?pvs=21) では全結合層の後に softmax 関数があります。これによって、出力が語彙数分の確率になり、最も大きいものを Transformer の出力単語とすることができます。

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, max_sequence_len, d_model, n_blocks, n_heads, d_k, d_v, d_ff
        )
        self.decoder = Decoder(
            tgt_vocab_size, max_sequence_len, d_model, n_blocks, n_heads, d_k, d_v, d_ff
        )
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        src_tgt_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_output = self.encoder(src, mask=src_mask)
        decoder_output = self.decoder(
            tgt, encoder_output, mask=tgt_mask, src_tgt_mask=src_tgt_mask
        )
        output = self.linear(decoder_output)
        return output
```

次に、Transformer を用いて1つの文章を出力する部分を作成しましょう。デコーダの説明で述べた通り、出力の計算は、エンコーダの出力とデコーダのそれまでの出力を入力として次の単語を出力します。この計算の実装を [<コード: Transformer 出力の計算>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) に示します▲注▲。

```python
    @torch.inference_mode
    def inference(self, src: Tensor, bos_token: int, eos_token: int) -> Tensor:
        tgt_tokens = torch.tensor([[bos_token]]).to(src.device)

        encoder_output = self.encoder(src)
        for _ in range(20):
            decoder_output = self.decoder(tgt_tokens, encoder_output)
            pred = self.linear(decoder_output)
            pred = torch.tensor([[pred[0, -1].argmax().item()]]).to(src.device)
            tgt_tokens = torch.cat((tgt_tokens, pred), axis=-1)
            if pred[0, 0].item() == eos_token:
                break

        return tgt_tokens

```

- ▲注▲
    
     このメソッドは Transformer クラス内に作成しています。
    

[<コード: Transformer 出力の計算>](https://www.notion.so/2-4-Transformer-34d85f8e7d2147509851e6f5117719fa?pvs=21) で実装している内容は以下のとおりです。

1. <bos> トークンをデコーダの入力 ( `tgt_tokens` ) に設定する
2. エンコーダの出力は for 文の前に計算して `encoder_output` に格納する
3. <eos> トークンが出力されるか、出力の限界 ( `max_tokens` ) になるまでループして1語ずつ推論する
    1. デコーダと線形層の計算
    2. 何番目の出力が最大であるかを取得
    3. これまでの出力結果 ( `tgt_tokens` ) に結合します。

`@torch.inference_mode` はこのメソッドの計算は推論として行うため、学習時に必要な値の保存等はしないことを表しています。Transformer が適切に学習されていると、以上のような計算によって適切な文章が生成されます。
# 4.4

4.3 節では大規模言語モデルの全パラメータを更新するような事前学習について学びました。このような学習はモデルの規模が大きくなるにつれて、ローカル環境やクラウド上の低コストなインスタンスでは計算リソースが足りない場合が多くなります。大企業や研究機関が大規模な事前学習や、5章で学ぶアラインメントを行ったモデルに対して微調整を加えたい場合でも、全パラメータの調整をするのは難しいです。本節ではこのような場合に対処するパラメータ効率の良いファインチューニング (PEFT; Parameter Effective Fine-Tuning) について学びます。

# 4.4.1 LLM の計算リソース

大規模モデルの利点として、少量のデータでもゼロショット学習や few-shot 学習が可能であることが挙げられます。これは、事前学習によって獲得された広範な言語理解能力や5章で学ぶアラインメントによって獲得された指示追従能力を利用することで、新しいタスクに迅速に適応できるからです。一方で、特定のドメインに特化させるなど、大規模言語モデルの能力を大幅にアップデートする目的で、モデルのパラメータを更新したい場合もあります。

近年の LLM  のような大きなモデルの場合、パラメータ数が膨大であるため、わずかなデータ数であっても学習のために大きな計算リソースが必要となります。そのため、限られた計算リソースで学習する方法が必要になります。具体的には、PEFT とよばれる学習するパラメータ数を少なくするような手法が取られます。以降で説明するLoRA などはその類の手法の中でも近年で特に人気の手法となっています。

この章では、4.4.2 で効率的な学習方法全般について概説し、4.4.3 で LoRA に関する解説を行います。その後、4.4.4 では LoRA のイメージを掴むためにイチから実装します。読者には、難解そうな学習方法がいかにシンプルに実装されているかを体験していただきたく思います。

# 4.4.2 効率的な学習方法

ファインチューニングには、モデルのすべてのレイヤーを調整する方法と、モデルパラメーターの一部のみを効率的に調整する PEFT（Parameter-Efficient Fine-Tuning）**[8]** という手法があります。PEFTは一般的に低コストで調整できるため、多様な手法が提案されています。

PEFT には大きく三つの分類があります **[7]**。「Additive methods（追加的手法）」「Reparametrization-based methods（再パラメータ化に基づいた手法）」「Selective methods（選択的手法）」です（▲図5▲）。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/3a95c661-8f02-4cc1-b951-2d59e1d5c236/Untitled.png)

※Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuningから引用

Additive methods は、既存のモデルに新たにパラメーターやレイヤーを追加し、その追加部分のみを学習します。これにはさらに、モデルの重みを固定したままで入力の埋め込みに学習可能なパラメータを追加する「Soft Prompts methods（ソフトプロンプト手法）」と、モデルに新たな学習可能なレイヤーを追加する「Adapters methods（アダプター手法）」の 2 種類があります。Soft Prompts は、入力トークンの埋め込みに学習可能なテンソルを連結し、バックプロパゲーションで最適化することでタスクの性能を上げる手法です。つまり、入力テキストにテンソルを加えて学習可能にします。Adapters は、Transformer のサブレイヤー後に小さな全結合ネットワークを導入します。分かりやすく言うと、モデルに層を追加して学習します。

次に、Reparametrization-based methods は低ランク表現を活用し、学習するパラメータ数を最小限に抑えるのが特徴です。その中でも特に人気な手法なのが LoRA と呼ばれる手法です。LoRA は Low-Rank Adaptation の略で、学習すべきパラメータを削減（Low-Rank）し適用（adaptation）する学習方法を指します。LoRA の詳細についてはこの後の 4.4.3 で詳しく解説します。

最後のSelective methodsは選択的に学習を行う手法で、例えばモデルのバイアスのみをチューニングするなどがあります。

[6] https://arxiv.org/abs/2106.09685

[7] https://arxiv.org/abs/2303.15647

[8] https://rinna.co.jp/news/2023/05/20230507.html

# 4.4.3 LoRAの理解

### LoRAとは何か

LoRA（Low-Rank Adaptation）は、4.4.2 でも説明したように事前学習済みの大規模な言語モデルに対して、軽量なファインチューニングを行うための手法である PEFT の一種です。LoRA は、以下の図にあるように、事前学習済みモデルのパラメータ W を直接更新するのではなく、低ランクのアダプタ A, B を追加することで、タスク固有の適応を行います。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/7c876f7c-dbaf-4dc4-899b-9cfbca05f311/Untitled.png)

※LoRA: Low-Rank Adaptation of Large Language Modelsから引用

具体的には、事前学習済みモデルの各層に、次のようなアダプタを追加します。

$$
W_{a} = W_{0} + \Delta W 
$$

$$
\Delta W = BA
$$

ここで、 $W_{0}$ は事前学習済みモデルの重み行列、$\Delta W$ はアダプタ、$B \in \mathbb{R}^{d \times r}$ と $A \in \mathbb{R}^{r \times d}$ はそれぞれランク▲注▲が $r$ の行列です。$d$ は元のモデルの入力次元数、$r$ はハイパーパラメータで、通常は $d$ よりもはるかに小さな値に設定されます。

- ▲注▲
    
    数学に関する補足を参照
    

ファインチューニングの際には、事前学習済みモデルのパラメータ $W_{0}$ は固定し、アダプタのパラメータ $B$ と $A$ のみを更新します。これにより、事前学習済みモデルの知識を保持しつつ、タスク固有の適応を効率的に行うことができます。

この近似によりどれだけのパラメータが削減できるか考えてみましょう。*A=100* および *B=500 とすると、 ΔW* のサイズは *100 × 500 = 50,000* になります。ここで、これを 2 つの小さな行列、*100×5* 次元行列 *W A* と *5×500* 次元行列 *W B* に分解するとします。これら 2 つの行列には、合計で *5 × 100 + 5 × 500 = 3,000 個のパラメータしかありません。* 

### LoRAの特徴と利点

LoRAには以下のような特徴と利点があります。

- 軽量なファインチューニング：LoRAでは、事前学習済みモデルのパラメータを直接更新するのではなく、低ランク行列に分解したアダプタのみを更新するため、ファインチューニングに必要なパラメータ数が大幅に削減されます。これにより、メモリ使用量や計算コストを抑えつつ、効率的にタスク固有の適応を行うことができます。
- 事前学習済みモデルの知識の保持：LoRAでは、事前学習済みモデルのパラメータを固定するため、事前学習で得られた汎用的な知識を保持したまま、タスク固有の適応を行うことができます。これにより、少量のデータでも効果的にファインチューニングを行うことができます。
- タスク間の干渉の軽減：LoRAでは、各タスクごとに独立したアダプタを追加するため、異なるタスク間での干渉を軽減することができます。これにより、マルチタスク学習や継続学習における負の転移の問題を緩和することができます。
- 柔軟性と拡張性：LoRAは、様々なアーキテクチャの事前学習済みモデルに適用することができ、また、アダプタのランクを調整することで、タスクの複雑さに応じて適応の度合いを制御することができます。これにより、幅広いタスクや要件に対して柔軟に対応することができます。LoRAによる重み更新 ΔW = BA の大きさを調整する係数としてスケーリングファクターが用いられます。

以上のように、LoRAは、大規模な事前学習済みモデルを効率的かつ効果的にタスク固有の要件に適応させるための強力な手法であり、自然言語処理やコンピュータビジョンなど、様々な分野で活用されています。

# 4.4.4 LoRAの実装

### LoRAの実装

4.4.3の内容を実装に落とし込むと以下のようになります。

```python
 class LoRA(nn.Module):
    def __init__(self, original_weight, rank, alpha):
        super(LoRA, self).__init__()
        self.alpha = alpha
        self.rank = rank
        self.original_weight = original_weight
        self.A = nn.Parameter(torch.randn(original_weight.size(0), rank))
        self.B = nn.Parameter(torch.randn(rank, original_weight.size(1)))

    def forward(self):
		    # ΔWの部分
        low_rank_modification = self.A @ self.B
        
        # alphaを使って補正の影響をスケーリング
        modified_weight = self.original_weight + self.alpha * low_rank_modification
        return modified_weight
```

実装はかなりシンプルですが、数学的なコーディングに慣れてない方向けに解説します。プログラムの 12 行目の**`self.A @ self.B`**は、**`self.A`**と**`self.B`**という二つの行列の積を計算しており、数式の$BA$の部分です。

### LoRAのハイパーパラメータ

プログラム中に登場する、rank や alpha はハイパーパラメータですが改めて機能について説明します。

rank (LoRA の内部次元数):

- 低ランク行列 A, B の内部次元数を決定します。
- rank が大きいほど、LoRA の表現力が高くなります。つまり、元の重み行列 W の変化をより細かく近似できます。
- 一方で、rank が大きすぎると過学習のリスクがあり、また計算コストも高くなります。
- rank は通常、元の重み行列 W の次元数よりもはるかに小さい値に設定します（例: 512 次元の埋め込み層に対して rank=8 など）。
- rank を増やすことで精度が向上する場合もありますが、その効果は次第に頭打ちになる傾向があります。

alpha (scaling factor):

- LoRA による重み更新 ΔW = BA の大きさを調整する係数です。
- alphaが大きいほど、LoRA による重み更新の影響が大きくなります。つまり、事前学習済みの重みからの変化量が大きくなります。
- alpha が小さすぎると、LoRA の効果が限定的になります。逆に大きすぎると、事前学習済みの知識を破壊してしまうリスクがあります。
- デフォルトでは alpha=1 に設定されることが多いですが、タスクやモデルに応じて調整することで精度が向上する場合があります。
- rank と alpha の最適値の組み合わせは、タスクやモデルによって異なります。

### LoRAのGPTモデルへの適用

GPTモデルのようなトランスフォーマーベースのモデルにおいてのLoRAを適用する場合、特定の層、特に自己注意層（Self-Attention layers）が考えられます。論文[1] でもクエリ（Query）、キー（Key）、値（Value）の重み行列に低ランクの補正を加えることにより、特定のタスクに対する表現能力を高めたと研究報告がされています。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f32ca4cc-631d-41b4-b55a-d4b4b3d47037/ce6b22e4-b12a-488d-b00b-5be5f21401e9/Untitled.png)

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class SelfAttentionWithLoRA(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_rank, lora_alpha):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_q = LinearWithLoRA(nn.Linear(embed_dim, embed_dim), lora_rank, lora_alpha)
        self.W_k = LinearWithLoRA(nn.Linear(embed_dim, embed_dim), lora_rank, lora_alpha)
        self.W_v = LinearWithLoRA(nn.Linear(embed_dim, embed_dim), lora_rank, lora_alpha)
        self.W_o = LinearWithLoRA(nn.Linear(embed_dim, embed_dim), lora_rank, lora_alpha)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.W_o(attn_output)
        
        return attn_output
```

この実装では、`LoRALayer`と`LinearWithLoRA`は前の例と同じです。`SelfAttentionWithLoRA`クラスは、標準的なself-attentionの実装に似ていますが、4つの重み行列(`W_q`, `W_k`, `W_v`, `W_o`)がLoRAを適用した`LinearWithLoRA`レイヤーに置き換えられています。

`forward`関数では、入力`x`にLoRAを適用したlinear変換を適用して、query (`q`)、key (`k`)、value (`v`)を計算します。その後、通常のself-attentionと同様に、attention weightsを計算し、valueとの加重和を取ることでattention outputを求めます。最後に、attention outputにLoRAを適用した線形変換(`W_o`)を適用して、最終的な出力を得ます。

この`SelfAttentionWithLoRA`モジュールを、TransformerBlockやGPTモデルの中で通常のself-attentionの代わりに使用することで、LoRAを適用したモデルを構築できます。

この実装では、LoRALayer と LinearWithLoRA, SelfAttentionWithLoRA のクラス定義を実装しています。

LoRALayer では、LoRA の先述の LoRAクラス の実装にもあったように、基本的なレイヤーの定義をしています。具体的には、初期化関数において低ランク行列 A, B の生成およびスケーリングファクターである alpha の定義がされており、forwardメソッドにて、ΔW の計算が実装されています。

LinearWithLoRA では、標準の線形レイヤーに LoRA を組み合わせたものです。既存の線形レイヤーの出力に対して LoRA による補正を加えるアルゴリズムが実装されています。

SelfAttentionWithLoRAでは、LoRAを使用した自己注意機構を実装しています。初期化メソッドでは、原論文において最も表現能力が高くなった条件と同じように、W_q, W_k, W_v, W_oに対して低ランク行列を定義しています。forwardメソッドにおいては、自己注意機構のアルゴリズムである、クエリ、キー、バリューや注意重み、注意出力の計算が実装されています。

以下のプログラムでは、先ほど定義した LinearWithLoRA と SelfAttentionWithLoRA を用いた GPT2 モデルに LoRA を適用する関数が実装されています。

```python
# from transformers import models
def apply_lora_to_gpt2(model, rank, alpha):
		for name, module in model.named_modules():
		    if isinstance(module, nn.Linear):
		        lora_layer  = LinearWithLoRA(module, rank, alpha)
		        parent_name = '.'.join(name.split('.')[:-1])
		        child_name  = name.split('.')[-1] 
		        parent = model.get_submodule(parent_name) 
		        setattr(parent, child_name, lora_layer)
		        
		    if isinstance(module, models.gpt2.modeling_gpt2.GPT2Attention):
		        embed_dim = module.embed_dim
		        num_heads = module.num_heads
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, SelfAttentionWithLoRA(embed_dim, num_heads, rank, alpha))
            print(embed_dim, num_heads, parent_name, child_name)
```

手順としては、モデルのモジュール名一覧から、線形層のインスタンスに該当する部分に対して、先ほど定義した LinearWithLoRAクラスを新たなアトリビューションとして置き換えるようにして適用します。これは SelfAttentionWithLoRA クラスを GPT2 の自己注意機構モジュールである GPT2Attention と置き換える際も同様に行います。

以下のプログラムでは実際に GPT2 の学習可能パラメータ数が LoRA の適用前後でどの程度減少したのかを確かめています。

一番下の出力を見ると元の GPT2 の学習可能パラメータ数は 124,439,808 個（ 約 1.2 億個）だったのが、LoRA の適用後には 29,346,440 個（約 3 千万）へと約 23.58% 減少していることがわかります。

```python
# from transformers import GPT2LMHeadModel
# モデルの学習可能パラメータ数を数える関数
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
original_model = GPT2LMHeadModel.from_pretrained('gpt2')
original_params = count_trainable_parameters(original_model)

# GPT2モデルのパラメータを固定
for param in original_model.parameters():
    param.requires_grad = False

# LoRAを適用したGPT-2モデルを作成
lora_rank = 8
lora_alpha = 16
lora_model = apply_lora_to_gpt2(original_model, lora_rank, lora_alpha)

# LoRAを学習可能パラメータ数を計算
lora_params = count_trainable_parameters(lora_model)

# 結果を表示
print(f"オリジナルGPT-2の学習可能パラメータ数: {original_params}")
print(f"LoRA適用後のGPT-2の学習可能パラメータ数: {lora_params}")
print(f"削減したパラメータ数: {original_params - lora_params}")
print(f"パラメータ減少率: {(lora_params / original_params) * 100:.2f}%")

### 出力 ###
# オリジナルGPT-2の学習可能パラメータ数: 124439808
# LoRA適用後のGPT-2の学習可能パラメータ数: 29346440
# 削減したパラメータ数: 95093368
# パラメータ減少率: 23.58%
```

[1]https://arxiv.org/abs/2106.09685

https://lightning.ai/pages/community/tutorial/lora-llm/

https://zenn.dev/zenkigen_tech/articles/2023-05-kurihara
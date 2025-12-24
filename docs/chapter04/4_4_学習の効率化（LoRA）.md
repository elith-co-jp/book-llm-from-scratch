# 4.4 学習の効率化（LoRA）

4.3 節では大規模言語モデルの全パラメータを更新するような事前学習について学びました。このような学習はモデルの規模が大きくなるにつれて、ローカル環境やクラウド上の低コストなインスタンスでは計算リソースが足りない場合が多くなります。大企業や研究機関が大規模な事前学習や、5章で学ぶアラインメントを行ったモデルに対して微調整を加えたい場合でも、全パラメータの調整をするのは難しいです。本節ではこのような場合に対処するパラメータ効率の良いファインチューニング (PEFT; Parameter Effective Fine-Tuning) について学びます。

# 4.4.1 LLM の計算リソース

大規模モデルの利点として、少量のデータでもゼロショット学習や few-shot 学習が可能であることが挙げられます。これは、事前学習によって獲得された広範な言語理解能力や5章で学ぶアラインメントによって獲得された指示追従能力を利用することで、新しいタスクに迅速に適応できるからです。一方で、特定のドメインに特化させるなど、大規模言語モデルの能力を大幅にアップデートする目的で、モデルのパラメータを更新したい場合もあります。

近年の LLM  のような大きなモデルの場合、パラメータ数が膨大であるため、わずかなデータ数であっても学習のために大きな計算リソースが必要となります。そのため、限られた計算リソースで学習する方法が必要になります。具体的には、PEFT とよばれる学習するパラメータ数を少なくするような手法が取られます。以降で説明するLoRA などはその類の手法の中でも近年で特に人気の手法となっています。

この章では、4.4.2 で効率的な学習方法全般について概説し、4.4.3 で LoRA に関する解説を行います。その後、4.4.4 では LoRA のイメージを掴むためにイチから実装します。読者には、難解そうな学習方法がいかにシンプルに実装されているかを体験していただきたく思います。4.4.5 では、実際のプロジェクトで活用できる HuggingFace PEFT ライブラリを用いた実践的な LoRA 学習方法を解説し、4.4.6 では青空文庫コーパスを用いた実際の学習結果を示します。

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

# 4.4.5 HuggingFace PEFTを用いたLoRA学習

4.4.4節ではLoRAの仕組みを理解するためにスクラッチで実装しました。実際のプロジェクトでLoRAを活用する場合は、HuggingFace が提供する PEFT（Parameter-Efficient Fine-Tuning）ライブラリを使用することで、より簡潔かつ安定した実装が可能です。本節では、PEFTライブラリを用いてGPT-2モデルにLoRAを適用し、ファインチューニングを行う方法を解説します。

### PEFTライブラリのインストール

PEFTライブラリは pip でインストールできます。

```bash
pip install peft
```

### LoRA設定の定義

PEFTライブラリでは、`LoraConfig` クラスを使用してLoRAのハイパーパラメータを設定します。

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # タスクの種類（因果言語モデル）
    r=8,                            # LoRAのランク（低ランク行列の次元数）
    lora_alpha=32,                  # スケーリングファクター
    lora_dropout=0.1,               # ドロップアウト率
    target_modules=["c_attn", "c_proj"],  # LoRAを適用する層
)
```

主要なパラメータの説明：

| パラメータ | 説明 |
|-----------|------|
| `task_type` | タスクの種類。言語モデルの場合は `TaskType.CAUSAL_LM` |
| `r` | 低ランク行列の次元数。4.4.3節で説明した rank に相当 |
| `lora_alpha` | LoRAの出力をスケーリングする係数 |
| `lora_dropout` | LoRA層に適用するドロップアウト率 |
| `target_modules` | LoRAを適用する層の名前リスト |

`target_modules` には、モデル内の線形層の名前を指定します。GPT-2の場合、`c_attn`（Query/Key/Valueの射影）と `c_proj`（出力射影）が Attention 層の主要な線形層です。

### モデルへのLoRA適用

`get_peft_model` 関数を使用して、既存のモデルにLoRAを適用します。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

# ベースモデルの読み込み
model_name = "rinna/japanese-gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRAの適用
model = get_peft_model(model, lora_config)

# 学習可能パラメータ数の確認
model.print_trainable_parameters()
```

出力例：
```
trainable params: 1,179,648 || all params: 337,668,096 || trainable%: 0.35%
```

ベースモデルの約3.4億パラメータのうち、LoRAによって追加された約118万パラメータ（0.35%）のみが学習対象となります。4.4.4節のスクラッチ実装と比較して、PEFTライブラリではより少ないパラメータ数で効率的な学習が可能です。

### Trainerを用いた学習

HuggingFaceの `Trainer` クラスと組み合わせることで、通常のファインチューニングと同様の手順で学習できます。

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# データコレーターの設定
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LMなのでMLMは無効
)

# 学習設定
training_args = TrainingArguments(
    output_dir="./models/gpt2-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=100,
    save_strategy="epoch",
    fp16=True,
)

# Trainerの初期化と学習実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
```

LoRAの学習では、通常のファインチューニングよりも高い学習率（例: 1e-4〜3e-4）を使用することが推奨されます。これは、学習対象のパラメータ数が少ないため、より大きな更新幅でも安定して学習できるためです。

### LoRAアダプターの保存と読み込み

学習済みのLoRAアダプターは、ベースモデルとは別に保存・読み込みが可能です。

```python
# アダプターの保存
model.save_pretrained("./models/gpt2-lora-adapter")

# アダプターの読み込み
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
model = PeftModel.from_pretrained(base_model, "./models/gpt2-lora-adapter")
```

保存されるファイルは `adapter_model.safetensors`（約5MB）と `adapter_config.json` のみであり、ベースモデル全体（約1.3GB）を保存する必要がありません。これにより、複数のタスク用アダプターを効率的に管理できます。

### アダプターのマージ

推論時のオーバーヘッドを削減したい場合、LoRAアダプターをベースモデルにマージして単一のモデルにできます。

```python
# アダプターをベースモデルにマージ
merged_model = model.merge_and_unload()

# マージ後のモデルを保存
merged_model.save_pretrained("./models/gpt2-lora-merged")
```

マージ後のモデルは通常のTransformersモデルとして扱えるため、LoRAを意識せずに推論を実行できます。ただし、マージ後はアダプターの切り替えができなくなる点に注意が必要です。

### スクラッチ実装との比較

4.4.4節のスクラッチ実装とPEFTライブラリの違いを以下にまとめます。

| 項目 | スクラッチ実装 | PEFTライブラリ |
|------|--------------|---------------|
| 実装の複雑さ | 高い（各層への適用を自分で実装） | 低い（設定を渡すだけ） |
| 適用対象の柔軟性 | 高い（任意の層に適用可能） | 中程度（target_modulesで指定） |
| 保存・読み込み | 自前で実装が必要 | 組み込みサポート |
| Trainerとの統合 | 追加実装が必要 | シームレスに動作 |
| デバッグの容易さ | 高い（コードが手元にある） | 低い（内部実装の理解が必要） |

学習の目的でLoRAの仕組みを理解するにはスクラッチ実装が有効ですが、実際のプロジェクトではPEFTライブラリの使用を推奨します。

[9] https://huggingface.co/docs/peft

# 4.4.6 学習結果と考察

本節では、4.4.5 で説明した HuggingFace PEFT を用いて、青空文庫コーパスで rinna/japanese-gpt2-medium を LoRA ファインチューニングした実際の結果を示します。

### 学習設定

| 項目 | 値 |
|------|-----|
| ベースモデル | rinna/japanese-gpt2-medium |
| データセット | 青空文庫（約11万サンプル） |
| GPU | NVIDIA RTX 6000 Ada Generation × 1 |
| エポック数 | 3 |
| バッチサイズ | 4（勾配累積4ステップ） |
| 学習率 | 1e-4 |
| LoRA rank (r) | 8 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| 対象モジュール | c_attn, c_proj |

### 学習結果

LoRA による学習は約5時間で完了しました（3エポック、約2550ステップ）。

| 項目 | 値 |
|------|-----|
| 学習時間 | 約5時間（シングルGPU） |
| 学習可能パラメータ数 | 約118万（全体の0.35%） |
| アダプターサイズ | 約8.3MB |
| ベースモデルサイズ | 約1.4GB |

アダプターのみを保存することで、約1.4GBのベースモデル全体を保存する必要がなく、約8.3MBのアダプターファイルのみで済みます。これにより、複数のタスク向けアダプターを効率的に管理できます。

### 4.3節との比較

4.3節のフルファインチューニングと比較して、LoRAの特徴を確認します。

| 項目 | フルファインチューニング（4.3節） | LoRA（本節） |
|------|----------------------------------|--------------|
| 学習パラメータ数 | 約3.4億（100%） | 約118万（0.35%） |
| 保存サイズ | 約1.4GB（モデル全体） | 約8.3MB（アダプターのみ） |
| 学習時間（シングルGPU） | 約6時間 | 約5時間 |
| 学習率 | 5e-5 | 1e-4（高め） |

GPT-2 medium 規模のモデルでは、メモリ使用量や学習時間に大きな差は見られません。LoRAの真価が発揮されるのは、7B以上の大規模モデルにおいてです。大規模モデルでは、フルファインチューニングには複数のGPUとDeepSpeed ZeROなどのメモリ最適化が必須となりますが、LoRAを使用すればシングルGPUでも学習が可能になります。

### 生成サンプル

学習の進行に伴う生成品質の変化を確認します。

**学習開始時（step 0）:**
```
プロンプト: 吾輩は猫である。名前はまだ無い。
生成結果: 僕を騙そうとしているのだろうか、それとも偽りかもしれない。
ただ僕には猫ちゃんに恋をしているのかいないのかが分からないだけだ。
```

**学習中盤（step 850）:**
```
プロンプト: 吾輩は猫である。名前はまだ無い。
生成結果: 犬も一匹しか居ない。そして、猫といふのは、其れでも一つの
物種だな」「それなら、私は何が何であれ、此処へ来ることが好ましいぞ」
```

**学習終了時（step 2550）:**
```
プロンプト: 吾輩は猫である。名前はまだ無い。
生成結果: 」「猫なのに、何で猫なのか?」「猫が猫じゃなければならぬ
理由は無いわよ。」と彼女は言った、「猫と言う言葉があるけれども、
猫には人間のような動物らしい性質はないからね。
```

**その他の生成例（step 2550）:**
```
プロンプト: 明治時代の
生成結果: です。だから、ととは、同じものと思つてゐたのであらうが、
一向に別物だとは感じない。けれども、今となっては、どうして違うのか
解らなかつたが、今更ながら考えてみると、それと何ら変らないことが分る。

プロンプト: 先生は言った。「
生成結果: 」「私はあなたの心をつかむことはできません。」彼はいやが
うえにも、もうたぶん自分の心を見失っているように思われた。そして、
そのくせ、ほんとうに自分がいま生きているということを、ふと忘れて
いるような気もした。
```

学習が進むにつれて、青空文庫の文体（旧仮名遣いや文語調の表現）を学習していることが確認できます。

### まとめ

本節では、LoRA を用いた効率的なファインチューニングについて学びました。

- **パラメータ効率**: 全パラメータの0.35%のみを学習することで、ストレージ効率が大幅に向上
- **実用上の利点**: 複数タスク向けアダプターを容易に管理可能
- **スケーラビリティ**: 7B以上の大規模モデルでは、シングルGPUでの学習を可能にする

LoRA は、計算リソースが限られた環境でも大規模言語モデルをカスタマイズできる強力な手法です。次章では、これらの技術を活用したアラインメント（指示追従能力の獲得）について学びます。
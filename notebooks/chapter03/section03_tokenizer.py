# -*- coding: utf-8 -*-
"""
第3章第3節: Tokenizer実装
このファイルには、Tokenizerの概要、入力処理、Embedding、BPEアルゴリズムの実装が含まれています。
"""

import re
import torch
import torch.nn as nn
from collections import Counter, defaultdict


# =============================================================================
# 3.3.1 Tokenizerの概要
# =============================================================================

def tokenize_text(text):
    """
    単語単位のトークン化を行う関数
    正規表現を使って、単語とスペース・句読点に分割
    """
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


def preprocess_text(text):
    """
    入力テキストの正規化を行う関数
    小文字に変換し、句読点の前後にスペースを追加
    """
    # 小文字に変換
    text = text.lower()
    # 句読点の前後にスペースを追加
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    return text


def assign_token_ids(tokens, vocab):
    """
    トークンIDへの変換を行う関数
    予め定義された語彙を使用して、各トークンに対応する数値IDを割り当て
    """
    token_ids = [vocab[token] for token in tokens]
    return token_ids


def build_vocabulary(tokens, max_size=None, min_freq=1, special_tokens=None):
    """
    ボキャブラリーを構築する関数
    """
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>']

    # トークンの出現頻度を計算
    token_counts = Counter(tokens)

    # 出現頻度の高い順にトークンをソート
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # 最小出現頻度と最大語彙サイズで語彙を制限
    filtered_tokens = [token for token, freq in sorted_tokens if freq >= min_freq][:max_size]

    # 特殊トークンを追加
    vocabulary = {token: idx for idx, token in enumerate(special_tokens + filtered_tokens)}
    return vocabulary


# =============================================================================
# 3.3.2 Tokenizerの入力処理
# =============================================================================

def add_special_tokens(tokens, max_length):
    """
    特殊トークンを追加する関数
    """
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    padding_length = max_length - len(tokens)
    tokens += ['[PAD]'] * padding_length
    return tokens


def truncate_tokens(tokens, max_length):
    """
    最大入力長に合わせてトークンを切り詰める関数
    """
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + ['[SEP]']
    return tokens


# =============================================================================
# 3.3.3 Embedding
# =============================================================================

def create_input_embedding_example():
    """
    入力エンベディングの実装例
    """
    vocab_size = 10
    embedding_dim = 128

    # 入力エンベディング層の定義
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)

    # トークンIDのバッチ
    token_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    # 入力エンベディングの計算
    input_embeddings = embedding_layer(token_ids)
    print("Input embeddings shape:", input_embeddings.shape)
    
    return embedding_layer, input_embeddings


def create_position_embedding_example():
    """
    位置エンベディングの実装例
    """
    seq_length = 5
    embedding_dim = 128

    # 位置エンベディング層の定義
    position_embedding_layer = nn.Embedding(seq_length, embedding_dim)

    # 位置IDの生成
    position_ids = torch.arange(seq_length).unsqueeze(0)

    # 位置エンベディングの計算
    position_embeddings = position_embedding_layer(position_ids)
    print("Position embeddings shape:", position_embeddings.shape)
    
    return position_embedding_layer, position_embeddings


# =============================================================================
# 3.3.4 BPE (Byte-Pair Encoding) の実装
# =============================================================================

# 1. シンプルなトークナイザーの実装
def simple_tokenizer(text):
    """
    単語を空白で区切るシンプルなトークナイザー
    """
    tokens = text.split()
    return tokens


# 2. 文字ペアの頻度計算
def compute_pair_freqs(splits, word_freqs):
    """
    各単語内の文字ペアの出現頻度を計算する関数
    """
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


# 3. 最も頻度の高いペアの結合
def merge_pair(a, b, splits, word_freqs):
    """
    特定したペアをすべての単語に対して適用し、新しいトークンを作成する関数
    """
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


# 4. BPEトークナイザーのメインクラス
class BPETokenizer:
    """
    BPE (Byte-Pair Encoding) トークナイザーの実装
    """
    
    def __init__(self):
        self.vocab = []
        self.merges = {}
    
    def train(self, corpus, num_merges=10):
        """
        BPEトークナイザーの学習
        """
        # 単語頻度の計算
        word_freqs = defaultdict(int)
        for text in corpus:
            tokens = simple_tokenizer(text)
            for token in tokens:
                word_freqs[token] += 1

        # ベース語彙の作成
        alphabet = []
        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()

        # 単語の文字分割
        splits = {word: [c for c in word] for word in word_freqs.keys()}

        # 語彙の初期化
        self.vocab = [""] + alphabet.copy()
        self.merges = {}

        # BPE学習ループ
        for _ in range(num_merges):
            pair_freqs = compute_pair_freqs(splits, word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = merge_pair(*best_pair, splits, word_freqs)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])

    def tokenize(self, text):
        """
        学習したルールを用いて新しいテキストをトークナイズ
        """
        tokens = simple_tokenizer(text)
        splits = [list(word) for word in tokens]  # 各単語を文字に分割

        for pair, merge in self.merges.items():
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


# =============================================================================
# 使用例とデモンストレーション
# =============================================================================

def demonstrate_basic_tokenization():
    """
    基本的なトークン化のデモンストレーション
    """
    print("=== 基本的なトークン化のデモ ===")
    
    text = "I like apples. You like oranges, not apples."
    tokens = tokenize_text(text)
    print("Original text:", text)
    print("Tokens:", tokens)
    
    # 前処理とトークン化
    preprocessed_text = preprocess_text(text)
    tokens = tokenize_text(preprocessed_text)
    print("Preprocessed tokens:", tokens)
    
    # 語彙辞書の構築
    vocab = {'<PAD>': 0, '<UNK>': 1, 'i': 2, 'like': 3, 'apples': 4, '.': 5, 'you': 6, 'oranges': 7, ',': 8, 'not': 9}
    token_ids = assign_token_ids(tokens, vocab)
    print("Token IDs:", token_ids)


def demonstrate_special_tokens():
    """
    特殊トークンのデモンストレーション
    """
    print("\n=== 特殊トークンのデモ ===")
    
    special_tokens = ['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']
    vocab = {'[CLS]': 0, '[SEP]': 1, '[MASK]': 2, '[PAD]': 3, '[UNK]': 4, 'i': 5, 'like': 6, 'apples': 7, '.': 8}

    tokens = ['i', 'like', 'apples', '.']
    max_length = 10
    tokens_with_special = add_special_tokens(tokens, max_length)
    print("Tokens with special tokens:", tokens_with_special)

    # 切り詰めの例
    long_tokens = ['[CLS]', 'i', 'like', 'apples', '.', 'you', 'like', 'oranges', ',', 'not', 'apples', '.', '[SEP]']
    max_length = 10
    truncated_tokens = truncate_tokens(long_tokens, max_length)
    print("Truncated tokens:", truncated_tokens)


def demonstrate_embeddings():
    """
    エンベディングのデモンストレーション
    """
    print("\n=== エンベディングのデモ ===")
    
    # 入力エンベディング
    embedding_layer, input_embeddings = create_input_embedding_example()
    
    # 位置エンベディング
    position_embedding_layer, position_embeddings = create_position_embedding_example()


def demonstrate_bpe():
    """
    BPEアルゴリズムのデモンストレーション
    """
    print("\n=== BPEアルゴリズムのデモ ===")
    
    # コーパスの準備
    corpus = [
        "Large language models are transforming the landscape of natural language processing.",
        "Understanding tokenization is crucial for building efficient models.",
        "This chapter explores the intricacies of BPE and its impact on LLM performance.",
        "By mastering these techniques, you can enhance the capabilities of your models.",
    ]

    # BPEトークナイザーの学習
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.train(corpus, num_merges=10)
    
    print("Vocabulary:", bpe_tokenizer.vocab)
    print("Merges:", bpe_tokenizer.merges)

    # テストテキストのトークナイズ
    test_text = "Tokenization improves LLMs."
    result = bpe_tokenizer.tokenize(test_text)
    print("Original text:", test_text)
    print("BPE tokens:", result)


def main():
    """
    メイン関数：すべてのデモンストレーションを実行
    """
    demonstrate_basic_tokenization()
    demonstrate_special_tokens()
    demonstrate_embeddings()
    demonstrate_bpe()


if __name__ == "__main__":
    main()
"""GPT用のシンプルな文字レベルトークナイザ."""


class SimpleTokenizer:
    def __init__(self, text: str):
        """文字レベルのテキスト処理用トークナイザ.
        
        Args:
            text (str): 語彙を構築するためのテキストコーパス
        """
        # 一意な文字から語彙を構築
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
    
    def encode(self, text: str) -> list[int]:
        """テキストをトークンIDにエンコードする.
        
        Args:
            text (str): 入力テキスト文字列
            
        Returns:
            list[int]: トークンIDのリスト
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens) -> str:
        """トークンIDをテキストにデコードする.
        
        Args:
            tokens: トークンIDのリストまたはテンソル
            
        Returns:
            str: デコードされたテキスト文字列
        """
        return ''.join([self.idx_to_char.get(int(idx), '') for idx in tokens])
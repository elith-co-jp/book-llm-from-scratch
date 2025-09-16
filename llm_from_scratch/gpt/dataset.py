"""GPT学習用のデータセットとデータローディングユーティリティ."""

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer, block_size: int = 128):
        """自己回帰言語モデリング用のデータセット.
        
        Args:
            text (str): 入力テキストコーパス
            tokenizer: トークナイザインスタンス
            block_size (int): 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # テキスト全体をトークナイズ
        self.tokens = tokenizer.encode(text)
        print(f"データセットサイズ: {len(self.tokens)} トークン")
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 入力とターゲットシーケンスを取得
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_dataloaders(text: str, tokenizer, block_size: int = 128, batch_size: int = 64, 
                       train_split: float = 0.9, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """学習用と検証用のデータローダーを作成する.
    
    Args:
        text (str): 入力テキストコーパス
        tokenizer: トークナイザインスタンス
        block_size (int): 最大シーケンス長
        batch_size (int): 学習用バッチサイズ
        train_split (float): 学習用データの割合
        num_workers (int): データローディングワーカー数
    
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)のタプル
    """
    # データセットを作成
    dataset = TextDataset(text, tokenizer, block_size)
    
    # 学習用と検証用に分割
    n = len(dataset)
    n_train = int(train_split * n)
    n_val = n - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # データローダーを作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
"""GPTモデル用の学習ユーティリティ."""

import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class GPTTrainer:
    def __init__(self, model, train_loader, val_loader, 
                 learning_rate: float = 3e-4, weight_decay: float = 0.1, 
                 warmup_steps: int = 1000, max_steps: int = 10000,
                 grad_clip: float = 1.0, device=None):
        """GPTモデル用のトレーナークラス.
        
        Args:
            model: GPTモデルインスタンス
            train_loader: 学習用データローダー
            val_loader: 検証用データローダー
            learning_rate (float): 学習率
            weight_decay (float): AdamW用の重み減衰
            warmup_steps (int): ウォームアップステップ数
            max_steps (int): 最大学習ステップ数
            grad_clip (float): 勾配クリッピング値
            device: 学習を行うデバイス
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        
        # デバイスを設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self.model.to(self.device)
        
        # オプティマイザを設定
        self.optimizer = self.configure_optimizer(learning_rate, weight_decay)
        
        # スケジューラを設定
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=max_steps - warmup_steps
        )
        
    def configure_optimizer(self, learning_rate: float, weight_decay: float):
        """重み減衰付きAdamWオプティマイザを設定する."""
        # 重み減衰用にパラメータを分離
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_groups, lr=learning_rate, betas=(0.9, 0.95))
        return optimizer
    
    def get_lr(self, step: int) -> float:
        """ウォームアップ付きの学習率を計算する."""
        if step < self.warmup_steps:
            # リニアウォームアップ
            return self.optimizer.param_groups[0]['lr'] * step / self.warmup_steps
        else:
            # スケジューラを使用
            return self.optimizer.param_groups[0]['lr']
    
    @torch.no_grad()
    def evaluate(self, max_batches: int = 10) -> float:
        """検証セットでモデルを評価する."""
        self.model.eval()
        losses = []
        
        for i, (x, y) in enumerate(self.val_loader):
            if i >= max_batches:
                break
                
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            losses.append(loss.item())
        
        self.model.train()
        return np.mean(losses) if losses else float('inf')
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """単一の学習ステップ."""
        # 順伝播
        logits, loss = self.model(x, y)
        
        # 逆伝播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # オプティマイザステップ
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, log_interval: int = 100, eval_interval: int = 500) -> dict:
        """メイン学習ループ.
        
        Args:
            log_interval (int): ログ出力間隔ステップ数
            eval_interval (int): 評価間隔ステップ数
            
        Returns:
            dict: 学習損失と検証損失を含む辞書
        """
        self.model.train()
        
        train_losses = []
        val_losses = []
        
        step = 0
        epoch = 0
        
        print(f"学習デバイス: {self.device}")
        print(f"モデルパラメータ数: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
        start_time = time.time()
        
        while step < self.max_steps:
            epoch += 1
            
            for x, y in self.train_loader:
                if step >= self.max_steps:
                    break
                
                x, y = x.to(self.device), y.to(self.device)
                
                # 学習ステップ
                loss = self.train_step(x, y)
                train_losses.append(loss)
                
                # 学習率を更新
                if step >= self.warmup_steps:
                    self.scheduler.step()
                
                # ログ出力
                if step % log_interval == 0:
                    avg_loss = np.mean(train_losses[-log_interval:]) if len(train_losses) >= log_interval else loss
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{self.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {self.get_lr(step):.6f} | "
                          f"Time: {elapsed:.1f}s")
                
                # 評価
                if step % eval_interval == 0 and step > 0:
                    val_loss = self.evaluate()
                    val_losses.append(val_loss)
                    print(f"検証損失: {val_loss:.4f}")
                
                step += 1
        
        print(f"学習完了！総時間: {time.time() - start_time:.1f}s")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def save_checkpoint(self, path: str):
        """モデルチェックポイントを保存する."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(checkpoint, path)
        print(f"チェックポイントを{path}に保存しました")
    
    def load_checkpoint(self, path: str):
        """モデルチェックポイントを読み込む."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"{path}からチェックポイントを読み込みました")
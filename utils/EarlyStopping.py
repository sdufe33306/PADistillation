# utils/early_stopping.py

import torch

class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, verbose=False, path='checkpoint.pt'):
        """
        初始化早停机制。

        Args:
            patience (int): 在验证损失没有改善的epoch数之后停止训练。
            delta (float): 验证损失需要改善的最小变化量。
            verbose (bool): 是否打印相关信息。
            path (str): 保存最佳模型的路径。
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, logger):
        score = -val_loss  # 因为我们希望最小化损失

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, logger):
        """
        保存验证损失下降时的模型。

        Args:
            val_loss (float): 当前的验证损失。
            model (torch.nn.Module): 模型。
            logger (logging.Logger): 日志记录器。
        """
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

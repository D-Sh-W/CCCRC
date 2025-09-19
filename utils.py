import seaborn as sns
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class DynamicWeightedBCELoss(nn.Module):
    def __init__(self,
                 pos_weight_start=1.0,
                 pos_weight_max=5.0,
                 growth_rate=0.1,
                 gamma_pos=2.0,
                 gamma_neg=1.0,
                 hard_neg_ratio=0.3,
                 hard_pos_ratio=0.3,
                 alpha_pos=1.0,
                 alpha_neg=0.5,
                 dynamic_alpha=True,
                 epsilon=1e-7):
        super(DynamicWeightedBCELoss, self).__init__()
        self.pos_weight = float(pos_weight_start)
        self.pos_weight_max = float(pos_weight_max)
        self.growth_rate = float(growth_rate)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.hard_neg_ratio = hard_neg_ratio
        self.hard_pos_ratio = hard_pos_ratio
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.dynamic_alpha = dynamic_alpha
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, epoch=None):
        probs = torch.sigmoid(inputs).clamp(self.epsilon, 1. - self.epsilon)
        pt = probs * targets + (1 - probs) * (1 - targets)
        bce_loss = self.bce(inputs, targets)

        gamma = torch.where(targets == 1, self.gamma_pos, self.gamma_neg)
        alpha = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        focal_factor = (1 - pt) ** gamma
        loss = alpha * focal_factor * bce_loss

        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight
        loss = weights * loss

        neg_mask = (targets == 0)
        if neg_mask.any() and self.hard_neg_ratio > 0:
            neg_loss = loss[neg_mask]
            k = max(1, int(len(neg_loss) * self.hard_neg_ratio))
            topk_neg, neg_indices = torch.topk(neg_loss, k)
            hard_neg_mask = torch.zeros_like(targets, dtype=torch.bool)
            hard_neg_mask[neg_mask.nonzero().reshape(-1)[neg_indices]] = True
        else:
            hard_neg_mask = neg_mask

        pos_mask = (targets == 1)
        if pos_mask.any() and self.hard_pos_ratio > 0:
            pos_loss = loss[pos_mask]
            k = max(1, int(len(pos_loss) * self.hard_pos_ratio))
            topk_pos, pos_indices = torch.topk(pos_loss, k)
            hard_pos_mask = torch.zeros_like(targets, dtype=torch.bool)
            hard_pos_mask[pos_mask.nonzero().reshape(-1)[pos_indices]] = True
        else:
            hard_pos_mask = pos_mask

        selected_mask = hard_neg_mask | hard_pos_mask
        final_loss = loss[selected_mask]

        if self.dynamic_alpha and epoch is not None:
            alpha_t = min(1.0, epoch / 10.0)
            final_loss = alpha_t * final_loss + (1 - alpha_t) * bce_loss[selected_mask]
        else:
            final_loss = final_loss

        if epoch is not None and self.pos_weight < self.pos_weight_max:
            self.pos_weight = min(self.pos_weight_max, self.pos_weight * (1 + self.growth_rate))

        return final_loss.mean()


class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_save = False
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.should_save = True
            self.counter = 0
        else:
            self.should_save = False
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True


def save_confusion_matrix(cm, epoch, phase, save_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"{phase} Confusion Matrix - Epoch {epoch + 1}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{phase.lower()}_cm_epoch_{epoch + 1}.png"))
    plt.close()


def save_test_confusion_matrix(cm, phase, save_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"{phase} Confusion Matrix - Test")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{phase.lower()}_cm.png"))
    plt.close()


def save_test_PR_AUC(recall_curve, precision_curve, pr_auc, save_dir):
    plt.figure()
    plt.plot(recall_curve, precision_curve, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "PR_AUC.png"))
    plt.close()


def save_best_epoch_info(path, epoch, train_info, val_info):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"best_epoch = {epoch + 1}\n")
        f.write(f"train_acc  = {train_info[0]:.4f}\n")
        f.write(f"train_rec  = {train_info[1]:.4f}\n")
        f.write(f"train_fpr  = {train_info[2]:.4f}\n")
        f.write(f"train_loss = {train_info[3]:.4f}\n")
        f.write(f"val_acc    = {val_info[0]:.4f}\n")
        f.write(f"val_rec    = {val_info[1]:.4f}\n")
        f.write(f"val_fpr    = {val_info[2]:.4f}\n")
        f.write(f"val_loss   = {val_info[3]:.4f}")

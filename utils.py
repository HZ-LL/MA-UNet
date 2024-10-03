import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - (2. * (intersection + self.epsilon)) / (union + self.epsilon)
        return score

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    auc = 0
    specificity = 0
    dice_score = 0
    recall = 0
    precision = 0
    iou = 0

    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            target = target.to(device).unsqueeze(1)
            input = model(input)
            preds = torch.sigmoid(input)
            preds = (preds > 0.5).float()
            num_correct += (preds == target).sum()
            num_pixels += torch.numel(preds)
            tp += (preds * target).sum()
            tn += num_correct - tp
            fp += (preds - preds * target).sum()
            fn += (target - preds * target).sum()
            x = target.cpu().numpy()
            y = preds.cpu().numpy()
            xx = list(np.array(x).flatten())
            yy = list(np.array(y).flatten())
            auc = metrics.roc_auc_score(xx, yy, multi_class='ovo')
            precision += tp / ((tp + fp) + 1e-8)
            recall += tp / ((tp + fn) + 1e-8)
            specificity += tn / ((tn + fp) + 1e-8)
            dice_score += (2 * tp) / ((2 * tp + fp + fn) + 1e-8)
            iou += tp / ((tp + fp + fn) + 1e-8)

    acc = (num_correct / num_pixels * 100).cpu().numpy()
    dice = (dice_score / len(loader) * 100).cpu().numpy()
    iou = (iou / len(loader) * 100).cpu().numpy()
    precision = (precision / len(loader) * 100).cpu().numpy()
    recall = (recall / len(loader) * 100).cpu().numpy()
    specificity = (specificity / len(loader) * 100).cpu().numpy()
    print(f"Acc：{acc:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Recall:{recall:.4f}")
    print(f"Specificity:{specificity:.4f}")
    print(f"AUC:{auc * 100:.4f}")

    model.train()
    return acc, dice, iou, precision, recall, specificity, auc * 100


def save_preds_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (input, target) in enumerate(loader):
        input = input.to(device=device)
        target = target.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(input))
            preds = (preds > 0.5).float()

        stacked_images = torch.cat((target.unsqueeze(1), preds), dim=2)
        # 保存拼接后的图片
        torchvision.utils.save_image(stacked_images, f"{folder}/mask_pred_{idx}.png")

    model.train()
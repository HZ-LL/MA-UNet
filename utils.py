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

# 1
# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     tp = 0
#     fp = 0
#     fn = 0
#     tn = 0
#     auc = 0
#     specificity = 0
#     dice_score = 0
#     recall = 0
#     precision = 0
#     iou = 0

#     model.eval()
#     with torch.no_grad():
#         for input, target in loader:
#             input = input.to(device)
#             target = target.to(device).unsqueeze(1)
#             input = model(input)
#             preds = torch.sigmoid(input)
#             preds = (preds > 0.5).float()
#             num_correct += (preds == target).sum()
#             num_pixels += torch.numel(preds)
#             tp += (preds * target).sum()
#             tn += num_correct - tp
#             fp += (preds - preds * target).sum()
#             fn += (target - preds * target).sum()
#             x = target.cpu().numpy()
#             y = preds.cpu().numpy()
#             xx = list(np.array(x).flatten())
#             yy = list(np.array(y).flatten())
#             auc = metrics.roc_auc_score(xx, yy, multi_class='ovo')
#             precision += tp / ((tp + fp) + 1e-8)
#             recall += tp / ((tp + fn) + 1e-8)
#             specificity += tn / ((tn + fp) + 1e-8)
#             dice_score += (2 * tp) / ((2 * tp + fp + fn) + 1e-8)
#             iou += tp / ((tp + fp + fn) + 1e-8)

#     acc = (num_correct / num_pixels * 100).cpu().numpy()
#     dice = (dice_score / len(loader) * 100).cpu().numpy()
#     iou = (iou / len(loader) * 100).cpu().numpy()
#     precision = (precision / len(loader) * 100).cpu().numpy()
#     recall = (recall / len(loader) * 100).cpu().numpy()
#     specificity = (specificity / len(loader) * 100).cpu().numpy()
#     print(f"Acc：{acc:.4f}")
#     print(f"Dice: {dice:.4f}")
#     print(f"IoU: {iou:.4f}")
#     print(f"Precision:{precision:.4f}")
#     print(f"Recall:{recall:.4f}")
#     print(f"Specificity:{specificity:.4f}")
#     print(f"AUC:{auc * 100:.4f}")

#     model.train()
#     return acc, dice, iou, precision, recall, specificity, auc * 100

# 2
def check_accuracy(loader, model, device="cuda"):
    # 初始化指标计数器
    total_pixels = 0
    correct_pixels = 0
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假负例
    tn = 0  # 真负例
    preds_probs = []  # 存储预测概率
    targets_list = []  # 存储真实标签
    model.eval()
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            target = target.to(device).unsqueeze(1)
            preds = model(input)  #得到模型的输出
            preds_prob = torch.sigmoid(preds)  # 应用sigmoid得到预测概率
            preds_binary = (preds_prob > 0.5).float()  # 将预测概率二值化
            # 收集预测概率和真实标签用于AUC计算
            preds_probs.extend(preds_prob.cpu().numpy().flatten())
            targets_list.extend(target.cpu().numpy().flatten())

            # 其他指标
            total_pixels += torch.numel(preds_binary) #总像素
            correct_pixels += (preds_binary == target).sum().item() #预测正确像素

            tp += ((preds_binary == 1) & (target == 1)).sum().item()
            fp += ((preds_binary == 1) & (target == 0)).sum().item()
            fn += ((preds_binary == 0) & (target == 1)).sum().item()
            tn += ((preds_binary == 0) & (target == 0)).sum().item()
    # 计算各个指标
    accuracy = correct_pixels / total_pixels
    dice = (2 * tp) / ((2 * tp + fp + fn) + 1e-8)
    jaccard = tp / ((tp + fp + fn) + 1e-8)
    precision = tp / ((tp + fp) + 1e-8)
    recall = tp / ((tp + fn) + 1e-8)
    specificity = tn / ((tn + fp) + 1e-8)
    # 计算AUC
    # auc = roc_auc_score(targets_list, preds_probs)
    auc = 0
    # 输出结果
    print(f"Accuracy: {accuracy*100:.4f}%")
    print(f"Dice: {dice*100:.4f}%")
    print(f"Jaccard: {jaccard*100:.4f}%")
    print(f"Precision: {precision*100:.4f}%")
    print(f"Recall: {recall*100:.4f}%")
    print(f"Specificity: {specificity*100:.4f}%")
    print(f"AUC: {auc*100:.4f}%")
    model.train()
    return accuracy*100, dice*100, jaccard*100, precision*100, recall*100, specificity*100, auc

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

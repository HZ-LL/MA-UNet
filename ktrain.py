import torch
import torch.nn as nn
import pandas as pd
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch.optim as optim
from numpy import *
from MAunet import MyNet
# from network.UNet import UNet
from torch.utils.data import DataLoader
from dataset import GetDataset
from utils import check_accuracy, save_preds_imgs, DiceLoss

learning_rate = 1e-4
device = "cuda"
batch_size = 4
num_workers = 2
num_epochs = 100
image_height = 512
image_width = 512
flage = True
write = True
train_dir = "data/images"
train_mask_dir = "data/masks"
path = 'result/results.xlsx'
std_path = 'result/standard.xlsx'
losses = []

def Train(loader, model, optimizer, loss_bce, loss_dice, scaler, epoch):
    print("\n")
    print(f"------------------------Epoch:{epoch}------------------------")
    loop = tqdm(loader)

    epoch_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss1 = loss_bce(predictions, targets)
            loss2 = loss_dice(predictions, targets)
            loss = loss1*0.3 + loss2*0.7

        #反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    # 将平均损失添加到losses列表中
    epoch_loss /= len(loader.dataset)
    losses.append(epoch_loss)


if __name__ == "__main__":
    # 数据增强
    transform = alb.Compose(
        [
            alb.Resize(height=image_height, width=image_width),
            alb.Rotate(limit=35, p=1.0),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.1),
            alb.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # 数据加载
    all_dataset = GetDataset(
        image_dir = train_dir,
        mask_dir = train_mask_dir,
        transform = transform,
    )
    # 设置五折交叉
    kfold = KFold(n_splits=5, shuffle=True)
    best_IoU = 0.0
    #每折的平均值
    k_acc = []
    k_dice = []
    k_IoU = []
    k_precision = []
    k_recall = []
    k_specificity = []
    k_auc = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_dataset)):
        print(f'****************************Fold:{fold}********************************')
        train_data = torch.utils.data.SequentialSampler(train_ids)
        val_data = torch.utils.data.SequentialSampler(val_ids)

        train_loader = DataLoader(
            all_dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = True,  # 数据是否加载到CUDA固定内存页中
            sampler = train_data,
            drop_last = True
        )
        val_loader = DataLoader(
            all_dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = True,
            sampler = val_data,
            drop_last = True
        )
        # 模型、损失函数、优化器、学习率调度器初始化
        model = MyNet(in_channels=3, out_channels=1).to(device)
        # model = UNet(in_channels = 3, out_channels = 1).to(device)
        bce_loss = nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # 定义余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
        #使用混合精度进行训练
        scaler = torch.cuda.amp.GradScaler()
        epoch_list = []
        acc_list = []
        dice_list = []
        IoU_list = []
        precision_list = []
        recall_list = []
        specificity_list = []
        auc_list = []
        for epoch in range(num_epochs):
            epoch_list.append(epoch)
            # 训练
            Train(train_loader, model, optimizer, bce_loss, dice_loss, scaler, epoch)
            # 周期完后更新学习率
            scheduler.step()
            # 验证指标结果
            Acc, Dice, IoU, Precision, Recall, Specificity, AUC = check_accuracy(val_loader, model, device=device)
            acc_list.append(Acc)
            dice_list.append(Dice)
            IoU_list.append(IoU)
            precision_list.append(Precision)
            recall_list.append(Recall)
            specificity_list.append(Specificity)
            auc_list.append(AUC)
            # 保存最优模型
            if IoU > best_IoU:
                best_IoU = IoU
                # 保存模型和优化器的状态到检查点文件中
                torch.save(model.state_dict(), 'params/weight_point.pth')
                # 保存验证图片
                save_preds_imgs(val_loader, model, folder="saved_images/", device=device)
                print(f"Found better model! Saving to weight_point.pth with IoU Index: {best_IoU}")
        print(f"\n<<<-------------------第{fold}折结束----------------->>>")
        print(f"Mean Accuracy:{mean(acc_list)}")
        print(f"Mean IoU:{mean(IoU_list)}")
        print(f"Mean Dice:{mean(dice_list)}\n")
        # 记录每一折的均值
        k_acc.append(mean(acc_list))
        k_IoU.append(mean(IoU_list))
        k_dice.append(mean(dice_list))
        k_precision.append(mean(precision_list))
        k_recall.append(mean(recall_list))
        k_specificity.append(mean(specificity_list))
        k_auc.append(mean(auc_list))
        # 只写入一个kfols的数据（100个）
        if write:
            df = pd.read_excel(path)
            if len(IoU_list) != len(df):
                print("新数据的长度与DataFrame的行数不匹配！")
            else:
                # 将新数据添加到DataFrame中作为新列
                df['MA'] = IoU_list
                df['MAloss'] = losses
                # 保存修改后的DataFrame到Excel文件
                df.to_excel(path, index=False)
                print("新的数据列已添加到Excel文件中。")
            write = False
        # 只做一个kfold的图
        if flage:
            # 绘制IoU Index
            plt.figure()
            plt.plot(epoch_list, IoU_list, label='IoU Index')
            plt.title("IoU Index over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Index Value")
            # 添加图例
            plt.legend()
            plt.grid(True)
            # 调整布局
            plt.tight_layout()
            plt.savefig("IoU_Index.png")
            plt.close()

            # 绘制loss图
            plt.figure()
            plt.plot(epoch_list, losses, label='loss')
            plt.title("Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Index Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("Loss.png")
            plt.close()
            flage = False
    # 打印五折交叉验证最终结果
    res = [mean(k_acc), mean(k_dice), mean(k_IoU), mean(k_precision), mean(k_recall), mean(k_specificity), mean(k_auc)]
    dff = pd.read_excel(std_path)
    if len(dff) != len(res):
        print("新数据的长度与DataFrame的行数不匹配！")
    else:
        dff['MA'] = res
        dff.to_excel(std_path, index=False)
    print("\n训练完成！！！输出最终结果--------->")
    print(f"Mean Acc：{mean(k_acc)}")
    print(f"Mean Dice: {mean(k_dice)}")
    print(f"Mean IoU: {mean(k_IoU)}")
    print(f"Mean Precision:{mean(k_precision)}")
    print(f"Mean Recall:{mean(k_recall)}")
    print(f"Mean Specificity:{mean(k_specificity)}")
    print(f"Mean AUC:{mean(k_auc)}")

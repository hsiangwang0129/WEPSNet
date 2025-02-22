import os
import argparse
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric


def find_latest_checkpoint(save_folder, model_name, dataset_name):
    """自動尋找最新的 Checkpoint"""
    latest_checkpoint = os.path.join(save_folder, f"{model_name}_{dataset_name}_latest.pth")
    if os.path.exists(latest_checkpoint):
        print(f"📥 找到最新 checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    return None


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    parser.add_argument('--model', type=str, default='fast_scnn', help='model name')
    parser.add_argument('--dataset', type=str, default='citys', help='dataset name')
    parser.add_argument('--base-size', type=int, default=1024, help='base image size')
    parser.add_argument('--crop-size', type=int, default=768, help='crop image size')
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=2, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save-folder', default='./weights', help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file')
    
    args = parser.parse_args()

    # 自動尋找最新的 Checkpoint
    if args.resume is None:
        latest_checkpoint = find_latest_checkpoint(args.save_folder, args.model, args.dataset)
        if latest_checkpoint:
            args.resume = latest_checkpoint

    args.best_pred = 0.0
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(args)
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        cudnn.benchmark = True

        # 資料增強與標準化
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # 建立訓練 & 驗證資料集
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)

        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

        # 創建模型
        self.model = get_fast_scnn(dataset=args.dataset, aux=False).to(self.device)

        # 優化器 & Loss
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=False, aux_weight=0.4).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs, iters_per_epoch=len(self.train_loader), power=0.9)

        # 評估指標
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
        self.start_epoch = 0

        # 如果有 checkpoint，載入模型
        if args.resume:
            self.load_checkpoint(args.resume)

    def load_checkpoint(self, checkpoint_path):
        """載入訓練進度"""
        if os.path.isfile(checkpoint_path):
            print(f'📥 載入 checkpoint: {checkpoint_path}')
            
            # ✅ 加入 weights_only=False，確保 PyTorch 正常載入
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_pred = checkpoint['best_pred']
            print(f'✅ 成功恢復訓練，從 Epoch {self.start_epoch} 繼續！')
        else:
            print(f'⚠️ 找不到 checkpoint: {checkpoint_path}，從頭開始訓練！')

    def train(self):
        """訓練模型"""
        cur_iters = 0
        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            for i, (images, targets) in enumerate(self.train_loader):
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                if cur_iters % 10 == 0:
                    print(f'Epoch [{epoch}/{self.args.epochs}] Iter [{i+1}/{len(self.train_loader)}] '
                          f'| lr: {cur_lr:.8f} | Loss: {loss.item():.4f}')

            # 儲存 checkpoint
            self.validation(epoch)

        print("✅ 訓練完成！")

    def validation(self, epoch):
        """驗證模型"""
        is_best = False
        self.metric.reset()
        self.model.eval()
        
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            outputs = self.model(image)
            pred = torch.argmax(outputs[0], 1).cpu().data.numpy()
            self.metric.update(pred, target.numpy())
            pixAcc, mIoU = self.metric.get()
            print(f'Epoch {epoch}, Sample {i+1}, Validation pixAcc: {pixAcc * 100:.3f}%, mIoU: {mIoU * 100:.3f}%')

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        save_checkpoint(self.model, self.optimizer, epoch, self.args, self.best_pred, is_best)


def save_checkpoint(model, optimizer, epoch, args, best_pred, is_best=False):
    """存儲最新和最佳 Checkpoint"""
    directory = args.save_folder
    os.makedirs(directory, exist_ok=True)

    latest_path = os.path.join(directory, f'{args.model}_{args.dataset}_latest.pth')
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_pred': best_pred}, latest_path)
    print(f'💾 已儲存最新模型: {latest_path}')

    if is_best:
        best_path = os.path.join(directory, f'{args.model}_{args.dataset}_best.pth')
        shutil.copyfile(latest_path, best_path)
        print(f'🏆 已儲存最佳模型: {best_path}')


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

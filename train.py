import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
from preprocess.dataset import LiveSpoofCelebDataset, FAS_BCE_Dataset
from models.DC_CDN import DC_CDN_Classifier
from losses.focal_loss import FocalLoss
from preprocess import transformsv2 as Tv2
import pandas as pd


def get_argparse():
    parser = argparse.ArgumentParser(description="Train DC_CDN model for face anti-spoofing")

    parser.add_argument('--train_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/train_label.txt")
    parser.add_argument('--val_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt")
    parser.add_argument('--root_dir', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof", help='Root directory of dataset')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_id', type=str, default=0)

    parser.add_argument('--save_path', type=str, default="runs/dc_cdn")
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str, help='last checkpoint')

    parser.add_argument('--input_size', type=int, default=256)
    args = parser.parse_args()
    return args


def train(args):
    train_csv = args.train_csv
    val_csv = args.val_csv
    root_dir = args.root_dir

    batch_size = args.batch_size
    num_epochs = args.epochs
    patience = args.patience
    lr = args.lr

    save_path = args.save_path
    resume = args.resume


    # --- Setup ----
    os.makedirs(save_path, exist_ok=True)
    count = len(os.listdir(os.path.join(save_path))) + 1
    save_path = os.path.join(save_path, f"train_{count}")
    os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)

    last_checkpoint_path = os.path.join(save_path, "weights", "last_dc_cdn_checkpoint.pt")
    best_checkpoint_path = os.path.join(save_path, "weights", "best_dc_cdn_classifier.pt")

    # --- Device ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    train_transform = Tv2.Compose([
        Tv2.Resize((args.input_size, args.input_size)),
        Tv2.RandomHorizontalFlip(p=0.5),
        # Tv2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        Tv2.RandomResizedCrop(size=args.input_size, scale=(0.75, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        # transforms.RandomAffine(10, translate=(0.1, 0.1)),   # affine transform (xoay, dịch)
        Tv2.ColorTrans(mode=0), # BGR to RGB
        Tv2.ToTensor(),
        Tv2.Normalize(
            mean=[0.485, 0.456, 0.406],    # chuẩn hóa (ImageNet mean/std)
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    test_transform = Tv2.Compose([
        Tv2.Resize((args.input_size, args.input_size)),
        Tv2.CenterCrop(args.input_size),
        Tv2.ColorTrans(mode=0), # BGR to RGB
        Tv2.ToTensor(),
        Tv2.Normalize(
            mean=[0.485, 0.456, 0.406],    # chuẩn hóa (ImageNet mean/std)
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_df = pd.read_csv(train_csv, usecols=['path', 'label'])
    val_df = pd.read_csv(val_csv, usecols=['path', 'label'])

    train_dataset = FAS_BCE_Dataset(dataframe=train_df, base_dir=root_dir, transform=train_transform)
    val_dataset = FAS_BCE_Dataset(dataframe=val_df, base_dir=root_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # --- Model ---
    model = DC_CDN_Classifier(device=device).to(device)

    # --- Loss & Optimizer ---
    # criterion = nn.BCELoss()
    criterion = FocalLoss(gamma=2, alpha=0.2, task_type='binary')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Bổ sung scheduler cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=0  # T_max = số epoch để quay về lr min
    )

    # load resume
    start_epoch = 1
    if resume and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        print(f"🔄 Resumed from epoch {checkpoint['epoch']} with val acc {best_val_acc:.2f}%")


    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))

    # --- EarlyStopping ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop = False

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs + 1):
        if early_stop:
            print("⏹️ Early stopping triggered.")
            break

        # === Train ===
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (preds > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total * 100

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Epoch {epoch}] Val"):
                imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

                preds = model(imgs)
                loss = criterion(preds, labels)

                val_loss += loss.item()
                predicted = (preds > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100

        # === TensorBoard log ===
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # === Print log ===
        print(f"\n[Epoch {epoch}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # === Save best model ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✅ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                early_stop = True
                print("⛔ Early stopping reached patience.")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'epochs_no_improve': epochs_no_improve
        }, last_checkpoint_path)
        print("💾 Last checkpoint saved.")
        
        # ✅ Cập nhật learning rate sau mỗi epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}] LR: {current_lr:.6f}")
        writer.add_scalar("LR", current_lr, epoch)
        scheduler.step()


if __name__ == "__main__":
    args = get_argparse()
    train(args)
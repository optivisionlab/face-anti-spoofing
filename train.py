import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LiveSpoofDataset, LiveSpoofCelebDataset
from models.DC_CDN import DC_CDN_Classifier

# --- Config ---
dir_root = "/u01/manhquang/liveness/dataset"
train_live = os.path.join(dir_root, "train", "live")
train_spoof = os.path.join(dir_root, "train", "spoof")
val_live = os.path.join(dir_root, "val", "live")
val_spoof = os.path.join(dir_root, "val", "spoof")

batch_size = 4
num_epochs = 10
patience = 5
lr = 1e-4

# --- Device ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


train_csv = "/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/train_label.txt"
val_csv = "/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt"
root_dir = "/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof"

train_dataset = LiveSpoofCelebDataset(df_csv_path=train_csv, root_dir=root_dir, transform=transform, name='Celeb Train')
val_dataset = LiveSpoofCelebDataset(df_csv_path=val_csv, root_dir=root_dir, transform=transform, name='Celeb Val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- Model ---
model = DC_CDN_Classifier(device=device).to(device)

# --- Loss & Optimizer ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- TensorBoard ---
writer = SummaryWriter(log_dir="runs/dc_cdn_train")

# --- EarlyStopping ---
best_val_acc = 0.0
epochs_no_improve = 0
early_stop = False

# --- Training Loop ---
for epoch in range(1, num_epochs + 1):
    if early_stop:
        print("⏹️ Early stopping triggered.")
        break

    # === Train ===
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for imgs, _, labels in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
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
        for imgs, _, labels in tqdm(val_loader, desc=f"[Epoch {epoch}] Val"):
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
        torch.save(model.state_dict(), "best_dc_cdn_classifier.pth")
        print(f"✅ Best model saved (Val Acc: {val_acc:.2f}%)")
    else:
        epochs_no_improve += 1
        print(f"⏳ No improvement ({epochs_no_improve}/{patience})")
        if epochs_no_improve >= patience:
            early_stop = True
            print("⛔ Early stopping reached patience.")

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
from preprocess.dataset import FAS_CE_Dataset
from models.utils import get_model, load_pretrain
from losses.focal_loss import FocalLoss
from losses.single_center_loss import SingleCenterLoss
from preprocess import transformsv2 as Tv2
import pandas as pd
import random
from metrics.misc import compute_acer, compute_accuracy, adjust_learning_rate


def get_argparse():
    parser = argparse.ArgumentParser(description="Train DC_CDN model for face anti-spoofing")

    parser.add_argument('--train_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/train_label.txt")
    parser.add_argument('--val_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt")
    parser.add_argument('--root_dir', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof", help='Root directory of dataset')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--optimizer', type=str, default='SGD', help='use SGD or AdamW')
    parser.add_argument('--save_path', type=str, default="runs/dc_cdn")
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str, help='last checkpoint')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--accumulate_step', default=1, type=int)
    parser.add_argument('--schedule', default='40,60', type=str, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--total_steps', default=0, type=int)
    parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs to adjust lr')

    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    parser.add_argument('--single_center_loss_weight', default=0.001, type=float)
    parser.add_argument('--arch', metavar='ARCH', type=str, help='model architecture')
    
    parser.add_argument('--aug_spoof', action='store_true', help='use aug image live to spoof') 
    parser.add_argument('--tf_ratio', default=0.5, type=float, help='set time frame raito')
    parser.add_argument('--random_frame', action='store_true', help='set random frame')
    
    args = parser.parse_args()
    return args


def train(args):
    seed_value = 240
    random.seed(seed_value)
    
    # --- Setup ----
    os.makedirs(args.save_path, exist_ok=True)
    count = len(os.listdir(os.path.join(args.save_path))) + 1
    save_path = os.path.join(args.save_path, f"train_{count}")
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
        Tv2.RandomResizedCrop(size=args.input_size, scale=(0.75, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        Tv2.ColorTrans(mode=0), # BGR to RGB
        Tv2.ToTensor(),
        Tv2.Normalize(
            mean=[0.485, 0.456, 0.406],    # chuẩn hóa (ImageNet mean/std)
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    test_transform = Tv2.Compose([
        Tv2.Resize((args.input_size, args.input_size)),
        Tv2.ColorTrans(mode=0), # BGR to RGB
        Tv2.ToTensor(),
        Tv2.Normalize(
            mean=[0.485, 0.456, 0.406],    # chuẩn hóa (ImageNet mean/std)
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_df = pd.read_csv(args.train_csv, usecols=['path', 'label'])
    val_df = pd.read_csv(args.val_csv, usecols=['path', 'label'])

    train_dataset = FAS_CE_Dataset(dataframe=train_df, base_dir=args.root_dir, transform=train_transform, 
                                   is_train=True, random_frame=args.random_frame, tf_ratio=args.tf_ratio, aug_spoof=args.aug_spoof)
    
    val_dataset = FAS_CE_Dataset(dataframe=val_df, base_dir=args.root_dir, transform=test_transform, 
                                 is_train=False, random_frame=args.random_frame, tf_ratio=args.tf_ratio, aug_spoof=args.aug_spoof)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    
    args.warmup_steps = len(train_loader) * args.warmup_epochs
    args.total_steps = len(train_loader) * args.num_epochs
    
    print(f'warmup_steps: {args.warmup_steps}')
    print(f'total_steps: {args.total_steps}')
    
    # --- Model ---
    # create model
    model = None
    print("=> creating model '{}', fp16:{}".format(args.arch, args.fp16))
    model = get_model(args.arch, args.num_classes, args.fp16)
    
    if args.pretrained:
        print("=====load pretrained=====")
        load_pretrain(args.pretrained, model)
        print("=====load pretrained done=====")
        
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    feats, _ = model(inputs)
    
    # --- Loss & Optimizer ---
    criterion_scloss = SingleCenterLoss(m=0.3, D=feats.shape[1], use_gpu=True) # logger.info(f'Create single center loss, features dim: {D}')
    criterion_fcloss = FocalLoss(gamma=2, alpha=0.2, task_type='multi-class', num_classes=args.num_classes)
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                    eps=1.0e-08, betas=[0.9, 0.999],
                                    lr=args.lr, weight_decay=args.weight_decay)

    # # Bổ sung scheduler cosine annealing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=10, eta_min=0  # T_max = số epoch để quay về lr min
    # )

    scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=400)
    
    # load resume
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
    for epoch in range(start_epoch, args.num_epochs):
        if early_stop:
            print("⏹️ Early stopping triggered.")
            break
        
        # === Train ===
        model.train()
        train_loss, train_acc_total, train_acer_total = 0.0, 0.0, 0.0

        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch + 1}] Train")):
            global_step = epoch * len(train_loader) + i

            lr = adjust_learning_rate(optimizer, global_step, epoch, args)
            
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            feats, preds = model(imgs)
            # loss_fcl = criterion_fcloss(preds, labels)
            loss_scl = criterion_scloss(feats, labels)
            # loss = loss_fcl + args.single_center_loss_weight * loss_scl
            loss = args.single_center_loss_weight * loss_scl
            loss /= args.accumulate_step
            # loss.backward()
            # optimizer.step()

            train_loss += loss.item()
            train_acc_total += compute_accuracy(pred=preds, target=labels)
            train_acer_total += compute_acer(preds=preds, target=labels)
 
            if args.fp16:
                scaler.scale(loss).backward()
                if (i + 1) % args.accumulate_step == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    scaler.step(optimizer)
                    scaler.update()
                pass
            else:
                loss.backward()
                if (i + 1) % args.accumulate_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    
            if (i + 1) % args.accumulate_step == 0:
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_acc_total / len(train_loader)
        train_acer = train_acer_total / len(train_loader)
        
        # === Validation ===
        model.eval()
        val_loss, val_acc_total, val_acer_total = 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Epoch {epoch + 1}] Val"):
                imgs, labels = imgs.to(device), labels.float().to(device)

                feats, preds  = model(imgs)
                # loss_fcl = criterion_fcloss(preds, labels)
                loss_scl = criterion_scloss(feats, labels)
                # loss = loss_fcl + args.single_center_loss_weight * loss_scl
                loss = args.single_center_loss_weight * loss_scl
                
                val_loss += loss.item()
                val_acc_total += compute_accuracy(pred=preds, target=labels)
                val_acer_total += compute_acer(preds=preds, target=labels)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_acc_total / len(val_loader)
        val_acer = val_acer_total / len(val_loader)

        # === TensorBoard log ===
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", train_acc, epoch + 1 )
        writer.add_scalar("Accuracy/Val", val_acc, epoch + 1)
        writer.add_scalar("ACER/Train", train_acer, epoch + 1)
        writer.add_scalar("ACER/Val", val_acer, epoch + 1)
        writer.add_scalar("lr", lr, epoch + 1)

        # === Print log ===
        print(f"\n[Epoch {epoch + 1}] \tlr: {lr:.6f}\n"
            f"Train Loss: {avg_train_loss:.4f} \tTrain Acc: {train_acc:.4f} \tTrain ACER: {train_acer:.4f}\n"
            f"Val Loss: {avg_val_loss:.4f} \tVal Acc: {val_acc:.4f} \tVal ACER: {val_acer:.4f}")

        # === Save best model ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✅ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                early_stop = True
                print("⛔ Early stopping reached patience.")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'epochs_no_improve': epochs_no_improve,
            'lr': lr
        }, last_checkpoint_path)
        print("💾 Last checkpoint saved.")
        
        # ✅ Cập nhật learning rate sau mỗi epoch
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"[Epoch {epoch}] LR: {current_lr:.6f}")
        # writer.add_scalar("LR", current_lr, epoch)
        # scheduler.step()


if __name__ == "__main__":
    args = get_argparse()
    train(args)
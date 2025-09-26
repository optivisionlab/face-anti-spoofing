import os, glob
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from preprocess.dataset import FAS_CE_Dataset
from models.utils import get_model
from utils.utils import get_argparse
from preprocess import transformsv2 as Tv2
import pandas as pd
from metrics.misc import compute_acer, compute_accuracy


def test(args):
    # --- Setup ----
    os.makedirs(args.save_path, exist_ok=True)
    count = len(glob.glob(os.path.join(args.save_path, "test*"))) + 1
    save_path = os.path.join(args.save_path, f"test_{count}")
    os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)

    # --- Device ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---

    test_transform = Tv2.Compose([
        Tv2.Resize((args.input_size, args.input_size)),
        Tv2.ColorTrans(mode=0), # BGR to RGB
        Tv2.ToTensor(),
        Tv2.Normalize(
            mean=[0.485, 0.456, 0.406],    # chuẩn hóa (ImageNet mean/std)
            std=[0.229, 0.224, 0.225]
        ),
    ])

    df_test = pd.read_csv(args.test_csv, usecols=['path', 'label'])
  
    test_dataset = FAS_CE_Dataset(dataframe=df_test, base_dir=args.root_dir, transform=test_transform, 
                                  is_train=False, random_frame=args.random_frame, tf_ratio=args.tf_ratio)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # --- Model ---
    # create model
    model = None
    print("=> creating model '{}', fp16:{}".format(args.arch, args.fp16))
    model = get_model(args.arch, args.num_classes, args.fp16)
    model.load_state_dict(torch.load(args.best_model, map_location=device))
    model.eval()
        
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    _, _ = model(inputs)
    print("load best model done!")
    
    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))


    # === Validation ===
    model.eval()
    val_acc_total, val_acer_total = 0.0, 0.0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Test data: "):
            imgs, labels = imgs.to(device), labels.to(device)

            _, preds  = model(imgs)
            val_acc_total += compute_accuracy(pred=preds, target=labels)
            val_acer_total += compute_acer(preds=preds, target=labels)

    avg_test_acc = val_acc_total / len(test_loader)
    avg_test_acer = val_acer_total / len(test_loader)

    writer.add_scalar("Accuracy/Test", avg_test_acc, 1 )
    writer.add_scalar("ACER/Test", avg_test_acer, 1)

    # === Print log ===
    print(f"Accuracy: {avg_test_acc:.4f}, ACER: {avg_test_acer:.4f}")

    
if __name__ == "__main__":
    args = get_argparse()
    test(args)
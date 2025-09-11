import torch
import argparse, os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.CDCN import Conv2d_cd, CDCNpp
from preprocess.datatrain import FaceAntiSpoofing_TrainDataset
from preprocess.dataval import FaceAntiSpoofing_ValDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils.utils import AvgrageMeter, performances_score_val
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
from preprocess.transforms import Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    
    kernel_filter_list =[
        [[1,0,0],[0,-1,0],[0,0,0]], 
        [[0,1,0],[0,-1,0],[0,0,0]], 
        [[0,0,1],[0,-1,0],[0,0,0]],
        [[0,0,0],[1,-1,0],[0,0,0]], 
        [[0,0,0],[0,-1,1],[0,0,0]],
        [[0,0,0],[0,-1,0],[1,0,0]], 
        [[0,0,0],[0,-1,0],[0,1,0]], 
        [[0,0,0],[0,-1,0],[0,0,1]]
    ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float32)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss


def process_on_batch(model, sample_batched, loss_absolute=None, loss_contra=None, optimizer=None, criterion_absolute_loss=None, criterion_contrastive_loss=None, device='cpu'):
    
    inputs = sample_batched['image_x'].to(device)          # [B, C, H, W]
    # string_name = sample_batched['string_name']        # list tên ảnh
    spoof_label = sample_batched['spoofing_label'].to(device)
    binary_mask = sample_batched['binary_mask'].to(device) # [B, H, W]
    optimizer.zero_grad()
    
    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
    absolute_loss = criterion_absolute_loss(map_x, binary_mask)
    contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
    loss = absolute_loss + contrastive_loss
    
    n = inputs.size(0)
    loss_absolute.update(absolute_loss.data, n)
    loss_contra.update(contrastive_loss.data, n)
    
    sum_map = map_x.sum(dim=(1, 2))  # shape: (32,)
    sum_mask = binary_mask.sum(dim=(1, 2)) # shape: (32,)
    sum_mask = torch.clamp(sum_mask, min=1) # # tránh chia cho 0
    map_score = sum_map / sum_mask
    map_score = torch.clamp(map_score, max=1.0) # nếu score > 1 thì set = 1

    return map_score.cpu().detach().numpy(), spoof_label.squeeze(1).cpu().detach().numpy(), loss, loss_absolute, loss_contra


def validate_on_epoch(epoch, model, dataloader_val, criterion_absolute_loss, criterion_contrastive_loss, optimizer=None, device='cpu'):
    loss_absolute_val = AvgrageMeter()
    loss_contra_val =  AvgrageMeter()
    
    model.eval()
    # avg_val_ACC, avg_val_APCER, avg_val_BPCER, avg_val_ACER, avg_val_loss_absolute, avg_val_loss_contra = [], [], [], [], [], []
    avg_val_loss_absolute, avg_val_loss_contra = [], []
    score_pred, score_truth = np.array([]), np.array([])
    
    with torch.no_grad():
        for sample_batched in tqdm(dataloader_val, desc=f"[Epoch {epoch + 1}] Val"):
            # Lấy batch input và mask
            map_score, spoof_label, _, loss_absolute, loss_contra = process_on_batch(
                model=model, sample_batched=sample_batched, 
                loss_absolute=loss_absolute_val, loss_contra=loss_contra_val, 
                criterion_absolute_loss=criterion_absolute_loss, 
                criterion_contrastive_loss=criterion_contrastive_loss,
                optimizer=optimizer,
                device=device
            )
            avg_val_loss_absolute.append(loss_absolute.avg.cpu().detach().numpy())
            avg_val_loss_contra.append(loss_contra.avg.cpu().detach().numpy())
            score_pred = np.concatenate((score_pred, map_score))
            score_truth = np.concatenate((score_truth, spoof_label))
    
    avg_val_ACC, avg_val_APCER, avg_val_BPCER, avg_val_ACER = performances_score_val(map_score_val=zip(score_pred, score_truth)) # performances val                    
    return {
        'avg_val_loss_absolute': np.mean(avg_val_loss_absolute),
        'avg_val_loss_contra': np.mean(avg_val_loss_contra),
        'avg_val_ACC': np.mean(avg_val_ACC),
        'avg_val_APCER': np.mean(avg_val_APCER),
        'avg_val_BPCER': np.mean(avg_val_BPCER),
        'avg_val_ACER': np.mean(avg_val_ACER)
    }


def trainining_on_epoch(epoch, model, dataloader_train, criterion_absolute_loss, criterion_contrastive_loss, optimizer, device='cpu'):
    loss_absolute_train = AvgrageMeter()
    loss_contra_train =  AvgrageMeter()

    # avg_train_ACC, avg_train_APCER, avg_train_BPCER, avg_train_ACER, avg_train_loss_absolute, avg_train_loss_contra = [], [], [], [], [], []
    avg_train_loss_absolute, avg_train_loss_contra = [], []
    score_pred, score_truth = np.array([]), np.array([])
    
    for sample_batched in tqdm(dataloader_train, desc=f"[Epoch {epoch + 1}] Train"):
        map_score, spoof_label, loss, loss_absolute, loss_contra = process_on_batch(
            model=model, 
            sample_batched=sample_batched, 
            loss_absolute=loss_absolute_train, loss_contra=loss_contra_train, 
            criterion_absolute_loss=criterion_absolute_loss, 
            criterion_contrastive_loss=criterion_contrastive_loss,
            optimizer=optimizer,
            device=device
        )
        
        score_pred = np.concatenate((score_pred, map_score))
        score_truth = np.concatenate((score_truth, spoof_label))
        avg_train_loss_absolute.append(loss_absolute.avg.cpu().detach().numpy())
        avg_train_loss_contra.append(loss_contra.avg.cpu().detach().numpy())

    avg_train_ACC, avg_train_APCER, avg_train_BPCER, avg_train_ACER = performances_score_val(map_score_val=zip(score_pred, score_truth))
    train_metrics = {
        'avg_train_loss_absolute': np.mean(avg_train_loss_absolute),
        'avg_train_loss_contra': np.mean(avg_train_loss_contra),
        'avg_train_ACC': np.mean(avg_train_ACC),
        'avg_train_APCER': np.mean(avg_train_APCER),
        'avg_train_BPCER': np.mean(avg_train_BPCER),
        'avg_train_ACER': np.mean(avg_train_ACER)
    }
    return train_metrics, model, loss, optimizer

# main function
def train_model(dir_root, file_train_csv_path, file_val_csv_path, args):

    os.makedirs(os.path.join(args.log), exist_ok=True)
    print("created folder : ", args.log)
    count = len(os.listdir(os.path.join(args.log))) + 1
    logs_save = os.path.join(args.log, f"train{count}", "logs")
    weights_save = os.path.join(args.log, f"train{count}", "weights")
    os.makedirs(logs_save, exist_ok=True)
    print("created folder : ", logs_save)
    os.makedirs(weights_save, exist_ok=True)
    print("created folder : ", weights_save)
    
    # --- Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    echo_batches = args.echo_batches
    
    # --- Save TensorBoard & Weights ---
    writer = SummaryWriter(log_dir=logs_save)
    best_checkpoint_path = os.path.join(weights_save, "best_weight.pt")
    last_checkpoint_path = os.path.join(weights_save, "last_weight.pt")
    
    print("==============START==============")

    print("==============LOAD DATASET==============")
    train_df = pd.read_csv(file_train_csv_path, usecols=['path', 'label'])
    val_df = pd.read_csv(file_val_csv_path, usecols=['path', 'label'])
    
    # load train data
    train_data = FaceAntiSpoofing_TrainDataset(
        dataframe=train_df,
        base_dir=dir_root,
        resize=(256, 256),
        size_mask=(32, 32),
        transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()])
    )
    
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_worker)
    
    # load val data
    val_data = FaceAntiSpoofing_ValDataset(
        dataframe=val_df,
        base_dir=dir_root,
        resize=(256, 256),
        size_mask=(32, 32),
        transform=transforms.Compose([Normaliztion(), ToTensor()])
    )
            
    dataloader_val = DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_worker)
    
    print("==============LOAD DATASET DONE==============")
    
    print("==============SETUP TRAIN==============")
    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')

    else:
        # build new model
        model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
        model = model.to(device)

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # # Bổ sung scheduler cosine annealing
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=10, eta_min=0  # T_max = số epoch để quay về lr min
        # )
    
    criterion_absolute_loss = nn.MSELoss().to(device)
    criterion_contrastive_loss = Contrast_depth_loss().to(device)
    
    ACER_save = 1.0
    epochs_no_improve = 0
    early_stop = False
    print("==============SETUP TRAIN DONE==============")
    
    print("==============START TRAIN==============")
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        if early_stop:
            print("⏹️ Early stopping triggered.")
            break
        
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma
        
        model.train()
        
        train_metrics, model, loss, optimizer = trainining_on_epoch(
            epoch=epoch, 
            model=model, 
            dataloader_train=dataloader_train, 
            criterion_absolute_loss=criterion_absolute_loss, 
            criterion_contrastive_loss=criterion_contrastive_loss, 
            optimizer=optimizer,
            device=device, 
        )
        
        loss.backward()
        optimizer.step()
        
        # VAL DATA
        vail_metrics = validate_on_epoch(
            epoch=epoch, 
            model=model, 
            dataloader_val=dataloader_val,  
            criterion_absolute_loss=criterion_absolute_loss, 
            criterion_contrastive_loss=criterion_contrastive_loss, 
            optimizer=optimizer,
            device=device
        )
        
        # save best weight
        if vail_metrics['avg_val_ACER'] < ACER_save:
            ACER_save = vail_metrics['avg_val_ACER']
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✅ Best model saved (Val Acc: {ACER_save:.4f}%)")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                early_stop = True
                print("⛔ Early stopping reached patience.")
        
        # save last weight
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_ACER': train_metrics['avg_train_ACER'],
            'val_ACER': vail_metrics['avg_val_ACER'],
            'epochs_no_improve': epochs_no_improve
        }, last_checkpoint_path)
        print("💾 Last checkpoint saved.")
        
        # scheduler
        scheduler.step()
        
        # OUTPUT
        print('epoch:%d, Performances Train:  Accuracy= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % 
              (epoch + 1, train_metrics['avg_train_ACC'], train_metrics['avg_train_APCER'], train_metrics['avg_train_BPCER'], train_metrics['avg_train_ACER']))
        
        print('epoch:%d, Loss Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f, Total Loss= %.4f \n' % 
              (epoch + 1, train_metrics['avg_train_loss_absolute'], train_metrics['avg_train_loss_contra'], (train_metrics['avg_train_loss_absolute']+ train_metrics['avg_train_loss_contra'])))
        
        print('epoch:%d, Performances Val:  Accuracy= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % 
              (epoch + 1, vail_metrics['avg_val_ACC'], vail_metrics['avg_val_APCER'], vail_metrics['avg_val_BPCER'], vail_metrics['avg_val_ACER']))
        
        print('epoch:%d, Loss Val:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f, Total Loss= %.4f \n' % 
              (epoch + 1, vail_metrics['avg_val_loss_absolute'], vail_metrics['avg_val_loss_contra'], (vail_metrics['avg_val_loss_absolute'] + vail_metrics['avg_val_loss_contra'])))
        
        # Tensorboard log
        
        writer.add_scalar("Accuracy/Train", train_metrics['avg_train_ACC'], epoch + 1)
        writer.add_scalar("APCER/Train", train_metrics['avg_train_APCER'], epoch + 1)
        writer.add_scalar("BPCER/Train", train_metrics['avg_train_BPCER'], epoch + 1)
        writer.add_scalar("ACER/Train", train_metrics['avg_train_ACER'], epoch + 1)
        writer.add_scalar("Absolute_Depth_loss/Train", train_metrics['avg_train_loss_absolute'], epoch + 1)
        writer.add_scalar("Contrastive_Depth_loss/Train", train_metrics['avg_train_loss_contra'], epoch + 1)
        writer.add_scalar("Loss/Train", (train_metrics['avg_train_loss_absolute']+ train_metrics['avg_train_loss_contra']), epoch + 1)
        
        writer.add_scalar("Accuracy/Val", vail_metrics['avg_val_ACC'], epoch + 1)
        writer.add_scalar("APCER/Val", vail_metrics['avg_val_APCER'], epoch + 1)
        writer.add_scalar("BPCER/Val", vail_metrics['avg_val_BPCER'], epoch + 1)
        writer.add_scalar("ACER/Val", vail_metrics['avg_val_ACER'], epoch + 1)
        writer.add_scalar("Absolute_Depth_loss/Val", vail_metrics['avg_val_loss_absolute'], epoch + 1)
        writer.add_scalar("Contrastive_Depth_loss/Val", vail_metrics['avg_val_loss_contra'], epoch + 1)
        writer.add_scalar("Loss/Train", (vail_metrics['avg_val_loss_absolute'] + vail_metrics['avg_val_loss_contra']), epoch + 1)

    print("==============TRAIN DONE==============")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--num_worker', type=int, default=2, help='number of worker')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # default=0.0001
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='initial weight decay')  # default=0.0001
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--batchsize', type=int, default=32, help='initial batchsize')  # default= 32
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--train_csv', type=str, default="CelebA_Spoof/metas/intra_test/train_label.txt")
    parser.add_argument('--val_csv', type=str, default="CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt")
    parser.add_argument('--root_dir', type=str, default="celeba-spoof/CelebA_Spoof_/CelebA_Spoof", help='Root directory of dataset')
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    
    train_model(dir_root=args.root_dir, file_train_csv_path=args.train_csv, file_val_csv_path=args.val_csv, args=args)
    pass

import torch
import argparse, os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.CDCN import Conv2d_cd, CDCNpp
from preprocess.datatrain import Spoofing_Train_Images_Custom, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from preprocess.dataval import Spoofing_Val_Images_Custom, Normaliztion_valtest, ToTensor_valtest
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils.utils import AvgrageMeter


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


def validate_batch_images(model, dataloader_val, epoch, log_dir):
    model.eval()
    map_score_list = []

    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader_val):
            # Lấy batch input và mask
            inputs = sample_batched['image_x'].cuda()          # [B, C, H, W]
            string_name = sample_batched['string_name']        # list tên ảnh
            binary_mask = sample_batched['binary_mask'].cuda() # [B, H, W]

            # Forward qua model (theo batch)
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(inputs)  
            # map_x shape: [B, 1, H, W] hoặc [B, H, W] tùy model

            # Tính score cho từng ảnh trong batch
            for b in range(inputs.size(0)):
                score_norm = torch.sum(map_x[b]) / torch.sum(binary_mask[b])
                map_score = score_norm.item()
                if map_score > 1:
                    map_score = 1.0

                # Lưu kết quả (tên ảnh + score)
                map_score_list.append(f"{string_name[b]} {map_score}\n")

    # Ghi kết quả ra file txt
    map_score_val_filename = os.path.join(log_dir, f"{log_dir}_map_score_val_{epoch+1}.txt")
    with open(map_score_val_filename, 'w') as file:
        file.writelines(map_score_list)

    # Lưu model checkpoint
    torch.save(model.state_dict(), os.path.join(log_dir, f"{log_dir}_{epoch+1}.pt"))

    print(f"Validation done at epoch {epoch+1}, results saved in {map_score_val_filename}")


# main function
def train_test(dir_root, file_train_csv_path, file_val_csv_path, args):

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
        
    echo_batches = args.echo_batches

    print("==============START==============")


    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')

    else:

        model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
        model = model.cuda()

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    
    ACER_save = 1.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        model.train()
        
        # load train data
        train_data = Spoofing_Train_Images_Custom(
            csv_path=file_train_csv_path, 
            root_dir=dir_root,
            transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()])
            )
        
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
        
        # load val data
        val_data = Spoofing_Val_Images_Custom(
                    csv_path=file_val_csv_path, 
                    root_dir=dir_root, 
                    transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()])
                )
                
        dataloader_val = DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=4)
        
        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 

            optimizer.zero_grad()

            # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
            
            #pdb.set_trace()
            #pdb.set_trace()
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            
            loss =  absolute_loss + contrastive_loss
            
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            
            
            if i % echo_batches == echo_batches - 1:    # print every 50 mini-batches
                
                # visualization
                # FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
        
            #break            
            
        # whole epoch average
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))

        validate_batch_images(model, dataloader_val, epoch, args.log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.00008, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_BinaryMask_P1_07", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--train_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/train_label.txt")
    parser.add_argument('--val_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt")
    parser.add_argument('--root_dir', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof", help='Root directory of dataset')
    args = parser.parse_args()
    train_test(dir_root=args.root_dir, file_train_csv_path=args.train_csv, file_val_csv_path=args.val_csv, args=args)
    pass

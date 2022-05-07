import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import glob
from configs.config import get_gonfig
from data.data_loader import train_loader
from data.RSA import RSA
from utils.loss import loss_func
from models.WaterSNet import WaterSNet
import os
from utils.log import get_logger,save_ckpt,summary_writer
from tqdm import tqdm
import torch.optim.lr_scheduler as lrs

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
#-------------------------------------get config and set exp name----------------------------------------
    args=get_gonfig()
    train(args)

def train(args):
    exp_name=args['exp_name']
    tensorboard_writer=summary_writer(exp_name)
    device=torch.device(args['Train']['device'] if torch.cuda.is_available() else "cpu")
    #--------------------------------------loading data------------------------------------------
    tra_img_name_list = glob.glob(args['Train']['tra_image_dir'] + '*' + '.jpg')
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        tra_lbl_name_list.append(args['Train']['tra_lbl_dir'] + imidx + '.png')
    logger = get_logger(exp_name)

    logger.info("---")
    logger.info(exp_name)
    logger.info("train images: {}".format(len(tra_img_name_list)))
    logger.info("train labels: {}".format(len(tra_lbl_name_list)))
    logger.info("---")
    print("-------------------")
    print("\033[1;33m"+exp_name+"\033[0m")
    print("train images: {}".format(len(tra_img_name_list)))
    print("train labels: {}".format(len(tra_lbl_name_list)))
    print("-------------------")
    train_num = len(tra_img_name_list)
    trainloader=train_loader(tra_img_name_list,tra_lbl_name_list,args['Train']['batch_size'],args['Train']['train_size'])
    # ----------------------------------define model --------------------------------------------
    net_name=args['Train']['net']
    net=WaterSNet()
    net.to(device)
    #------------------------------ define optimizer ------------------------------------------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=args['Train']['optimizer_lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    milestones = []
    for i in range(1, args['Train']['epoch_num'] + 1):
        if i % 200 == 0:
            milestones.append(i)
    scheduler = lrs.MultiStepLR(optimizer, milestones, 0.5)
    #======================================= training process ===================================================
    logger.info("---start training...")
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    ite_num4val = 0
    IMG_MEAN = np.array((0.41, 0.46, 0.48), dtype=np.float32)
    IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1, 3, 1, 1))
    current_epoch=args['Train']['current_epoch']
    # ----------------------------------------training stage----------------------------------------------------------
    for epoch in range(current_epoch, args['Train']['epoch_num']):
        net.train()   
        bar_format = '{desc}{percentage:3.0f}%|{bar}|{n_fmt:.5s}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
        loop=tqdm(enumerate(trainloader,1),total=len(trainloader),ncols=100,bar_format = bar_format)
        for i, data in loop:
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs,labels = data[0],data[1]
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            B = inputs.size(0)
            perm = torch.randperm(B)
            inputs_shuffling = inputs[perm]
            inputs_RSA = RSA(inputs, inputs_shuffling)
            inputs_v = Variable(inputs.to(device), requires_grad=False)
            inputs_RSA_v = Variable(inputs_RSA.to(device), requires_grad=False)
            labels_v = Variable(labels.to(device), requires_grad=False)

            optimizer.zero_grad()        
            output1= net(inputs_v)
            output2= net(inputs_RSA_v)

            loss1=loss_func(output1,labels_v)
            loss2=loss_func(output2,labels_v)
            loss_consist=loss_func(output1,output2.detach())
            loss=loss1+loss2+2*loss_consist            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()       
    #---------------------------------------logging info-----------------------------------------------------
            logger.info("%s  :[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (net_name,
            epoch + 1, args['Train']['epoch_num'], (i + 1) * args['Train']['batch_size'], train_num, ite_num, running_loss / ite_num4val)) 
            loop.set_description(f"{ net_name } : Epoch [{ epoch+1 }/{ args['Train']['epoch_num'] }]") 
            loop.set_postfix(train_loss=f'{running_loss / ite_num4val:.3f}')
    #---------------------------------------save checkpoint--------------------------------------------------
            if ite_num % 3000 == 0:  # 
                tensorboard_writer.add_scalar('Training_loss', running_loss / ite_num4val, ite_num)
                ckpt ={
                    "net": net.state_dict(),
                    "current_epoch": epoch,
                    "ite_num": ite_num
                    }
                save_ckpt(ckpt,exp_name,net_name=net_name,epoch=epoch,ite_num=ite_num)

                running_loss = 0.0
                ite_num4val = 0
            del loss
        scheduler.step()
        loop.close()
    #--------------------------------------last checkpoint-------------------------------------
    ckpt ={
        "net": net.state_dict(),
        "current_epoch": args['Train']['epoch_num']-1,
        "ite_num": ite_num
        }
    save_ckpt(ckpt,exp_name,net_name=net_name,epoch=args['Train']['epoch_num']-1,ite_num=ite_num)
    tensorboard_writer.close()
    print('\033[1;33m-------------Congratulations! Training Done!!!-------------\033[0m')
    logger.info('-------------Congratulations! Training Done!!!-------------')

if __name__ == '__main__':
    main()
from __future__ import print_function
from __future__ import division
import argparse
import os
import time

import torch
import torch.utils.data
import torch.optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.nn import  DataParallel
cudnn.benchmark = True
from utils.visualizer import *
from utils.view_model import *
import torchvision
from models.net import *
from data.dataset import ImageList
import lfw_eval
from  models.layer import *
from models.resnet import *
from models.metrics import *
from config.config import Config


opt = Config() 
flag_cuda = opt.use_gpu and torch.cuda.is_available()
common_dict = {}
#-------------------------------------tensorboard for pytorch--------------------------- 
try:
   from tensorboardX import SummaryWriter
   prefix = '%s-%s-%d ' % (opt.backbone,opt.metric,opt.tryt)
   summary_dir = os.path.join(prefix, "tf_summary")
   if os.path.exists(summary_dir):
        print ("Delete old summary in first.")
        os.system("rm -rf {}".format(summary_dir))
   common_dict["tensorboard_writer"] = SummaryWriter(summary_dir)
   common_dict["iterations"] = int(0)
   print ("Enable tensorboard summary.")
   print ("Please using 'python -m tensorboard.main --logdir={}'".format(summary_dir))
except Exception as ex:
   common_dict["tensorboard_writer"] = None
   print ("Disable tensorboard summary. please install tensorboardX in first.")
   print ("Easy to install by 'pip install tensorboardX --user'")

def main():
    # --------------------------------------model----------------------------------------
    if opt.display:
       visualizer = Visualizer()
    
    device = torch.device("cuda")
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    elif opt.backbone == 'sphere20':
        model = sphere20()
    elif opt.backbone == 'sphere20_bd':
        model = sphere20_bd(dropout_Probability=opt.dropout_Probability)
    elif opt.backbone == 'sphere36_bd':
        model = sphere36_bd(dropout_Probability=opt.dropout_Probability)    

    #---------------------------------------Metric Learning-------------------------------#
    if opt.metric == 'add_margin':
        metric_fc = MarginCosineProduct(512, opt.num_classes, s=30, m=0.40)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    elif opt.metric == 'softmax':
        metric_fc = nn.Linear(512, opt.num_classes)
    
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    #metric_fc = DataParallel(metric_fc)

    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
    model.module.save(opt.checkpoints_path+opt.metric+ '_0_checkpoint.pth')
    
    #-------------------------------Set mix or single-------------------------------------#
    filedict = {}
    filedict_val = {}
    if opt.mix_flag == True:
        if opt.val_flag == True:
            filedict[opt.train_root] = opt.train_list
            for i in range(opt.mix_nums):
                filedict[opt.mix_train[int(i)][0]] = opt.mix_train[int(i)][1]
            filedict_val[opt.train_root] = opt.val_list
            for i in range(opt.mix_nums):
                filedict[opt.mix_train[int(i)][0]] = opt.mix_val[int(i)][2]
        else:
            filedict[opt.train_root] = opt.train_all_list
            for i in range(opt.mix_nums):
                filedict[opt.mix_train[int(i)][0]] = opt.mix_train[int(i)][3]
            filedict_val[opt.train_root] = opt.val_list
    else:
        if opt.val_flag == True:
            filedict[opt.train_root] = opt.train_list
            filedict_val[opt.train_root] = opt.val_list
        else:
            filedict[opt.train_root] = opt.train_all_list      
    # ------------------------------------load image---------------------------------------
    train_loader = torch.utils.data.DataLoader(
        ImageList(filedict=filedict,
                  transform=transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                  ])),
        batch_size=opt.train_batch_size, shuffle=True,
        num_workers=opt.num_workers)
    if opt.val_flag == True:
        val_loader = torch.utils.data.DataLoader(
            ImageList(fileDict=filedict_val,
                      transform=transforms.Compose([
                      #transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
                     ])),
            batch_size=opt.train_batch_size, shuffle=False,num_workers=opt.num_workers)
    print('length of train Dataset: ' + str(len(train_loader.dataset)))
    print('Number of Classses: ' + str(opt.num_classes))

    # ----------------------------------------optimizer----------------------------------
    # MCP = layer.AngleLinear(512, args.num_class).cuda()
    # MCP = torch.nn.Linear(512, args.num_class, bias=False).cuda()
    if opt.restore_checkpoints != "None":
        print ("Resotre ckpt from {}".format(opt.restore_checkpoints))
        model.load_state_dict(torch.load(opt.restore_checkpoints))
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adm':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'mom':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=opt.lr,momentum=opt.mom,weight_decay=opt.weight_decay)

    # ----------------------------------------train----------------------------------------
    # lfw_eval.eval(args.save_path + 'CosFace_0_checkpoint.pth')
    for epoch in range(1, opt.max_epoch + 1):
        # scheduler.step()
        train(train_loader, model, metric_fc, criterion, optimizer, epoch)
        model.module.save(opt.checkpoints_path+opt.metric+'_' + str(epoch) + '_checkpoint.pth')
        lfw_eval.eval(common_dict,opt.checkpoints_path +opt.metric+'_' + str(epoch) + '_checkpoint.pth')
    print('Finished Training')


def train(train_loader, model, metric_fc, criterion, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0
    #time3 = 0    
    #start1 =time.time()
    #---------------------------------train------------------------------------------
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        #end1 = time.time()
        #if batch_idx >= 1:
        #    print("load data:", end1 - time3)
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, opt.step_size)
        lr = optimizer.param_groups[0]['lr']
        if flag_cuda:
            data, target = data.cuda(), target.cuda()
            #print("CUDA IS USING")
        data, target = Variable(data), Variable(target)
        # compute output
        #start2 = time.time()
        output = model(data)
        #end2 = time.time() 
        if opt.metric == 'softmax':
           output = metric_fc(output)
        else:
           output = metric_fc(output, target)
        loss = criterion(output, target)
        loss_display += loss.data[0]
        #print("preprocessing:", start2 - end1 )
        #print("train model:",end2-start2)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #time3 = time.time()
        #print("backward time:", time3 - end2)
        if batch_idx % opt.print_freq == 0:
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            target = target.data.cpu().numpy()
            train_acc = np.mean((output == target).astype(int))
            time_used = time.time() - time_curr
            loss_display /= opt.print_freq
            if opt.metric == 'sphere':
                INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(metric_fc.m, 0)
            elif opt.metric == 'softmax':
                INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(0, 0) 
            else:
                INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(metric_fc.m, metric_fc.s)
            
            # INFO = ' lambda: {:.4f}'.format(MCP.lamb)
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f},Acc: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display,train_acc, time_used, opt.print_freq) + INFO
            )
            time_curr = time.time()
            
            if opt.display:
                    visualizer.display_current_results(iteration, loss_display, name='train_loss')
                    visualizer.display_current_results(iteration, train_acc, name='train_acc')

            if opt.display_tx:
                    if common_dict["tensorboard_writer"] is not None:
                        common_dict["tensorboard_writer"].add_scalar("loss", loss_display,
                                                                    iteration)
                        common_dict["tensorboard_writer"].add_scalar("lr", lr,
                                                                    iteration)
                        common_dict["tensorboard_writer"].add_scalar("train_acc", train_acc,
                                                                    iteration)
            loss_display = 0.0
        common_dict['iterations'] += 1 
        if opt.val_flag == True:
            val(model,epoch,iteration,train_acc)

def val(model,epoch,iters,t_acc):
    v_start = time.time()
    model.eval()
    for batch_idx, (data, target) in enumerate(val_loader, 1): 
       if flag_cuda:
          data, target = data.cuda(), target.cuda()
       data, target = Variable(data), Variable(target)
       output = model(data)
       if opt.metric == 'softmax':
           output = metric_fc(output, target)
       else:
           output = metric_fc(output)
       output = output.data.cpu().numpy()
       output = np.argmax(output, axis=1)
       label = label.data.cpu().numpy()       
       v_acc += np.mean((output == target).astype(int))
    v_acc = v_acc/len(val_loader)
    print("len(valloader): ",len(val_loader))
    v_end = time.time()
    v_speed = v_end - v_start
    print('{} train epoch {} val iter {} {} iters/s acc {}'.format(time_str, epoch, iters, v_speed,v_acc))
    if opt.display:
       visualizer.display_current_val_results(iters, v_acc, name='val_acc')
        
    if opt.display_tx:
        if common_dict["tensorboard_writer"] is not None:
             common_dict["tensorboard_writer"].add_scalar("val_acc", v_acc,
                                                                    iters)        
             common_dict["tensorboard_writer"].add_scalars("group/acc", {"v_acc":v_acc,
                                                                         "t_acc":t_acc},iters)
def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = opt.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass

if __name__ == '__main__':
    main()

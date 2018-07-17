from PIL import Image
import numpy as np
from models.resnet import *
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from models.net import *
from config.config import Config

def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(common_dict,model_path=None):
    predicts = []
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    elif opt.backbone == 'sphere20':
        model = model = sphere20()
    elif opt.backbone == 'sphere20_bd':
        model = sphere20_bd(dropout_Probability=opt.dropout_Probability)
    elif opt.backbone == 'sphere36_bd':
        model = sphere36_bd(dropout_Probability=opt.dropout_Probability)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #root = '/mnt/lustre/jiangting/Hedlen/github/datasets/lfw-deepfunneled-align/'
    root = opt.lfw_root
    lfw_test_pair = opt.lfw_test_list
    with open(lfw_test_pair) as f:
        pairs_lines = f.readlines()[1:]
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    for i in range(6000):
        p = pairs_lines[i].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

        img1 = Image.open(root + name1).convert('RGB')
        img2 = Image.open(root + name2).convert('RGB')
        img1, img1_, img2, img2_ = transform(img1), transform(F.hflip(img1)), transform(img2), transform(F.hflip(img2))
        img1, img1_ = Variable(img1.unsqueeze(0).cuda(), volatile=True), Variable(img1_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        img2, img2_ = Variable(img2.unsqueeze(0).cuda(), volatile=True), Variable(img2_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        f1 = torch.cat((model(img1), model(img1_)), 1).data.cpu()[0]
        f2 = torch.cat((model(img2), model(img2_)), 1).data.cpu()[0]

        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    acc = np.mean(accuracy)
    if opt.display:
            visualizer.display_current_results(common_dict["iterations"], acc, name='lfw_acc')
        
    if opt.display_tx:
            if common_dict["tensorboard_writer"] is not None:
                common_dict["tensorboard_writer"].add_scalar("lfw_acc", acc,
                                                                    common_dict["iterations"])
    return np.mean(accuracy), predicts


if __name__ == '__main__':
    '''
    _, result = eval(model_path='checkpoint/SphereFace_24_checkpoint.pth')
    np.savetxt("result.txt", result, '%s')
    '''
    eval(model_path='checkpoint/CosFace_30_checkpoint.pth')
    '''
    for epoch in range(1, 31):
        eval('checkpoint/CosFace_' + str(epoch) + '_checkpoint.pth')
    '''

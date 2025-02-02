import builtins
import datetime
import os
import math
import sys
import time
import random
import argparse
import torch
import torch.nn 
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from plrc_current import PLRC
from utils.model import PaPi
from utils.resnet import *
from utils.utils_algo import *
from utils.utils_loss import PaPiLoss

from utils.cifar10 import load_noisy_cifar10
from utils.cifar100 import load_noisy_cifar100

parser = argparse.ArgumentParser(description='PyTorch implementation of PaPi (Towards Effective Visual Representations for Partial-Label Learning)')

parser.add_argument('--exp-type', default='rand', type=str, choices=['rand', 'ins'], help='Different exp-types')

parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'],
                    help='dataset name')

parser.add_argument('--exp-dir', default='./train', type=str,
                    help='experiment directory for saving checkpoints and logs')

parser.add_argument('--pmodel_path', default='./pmodel/cifar10.pt', type=str,
                    help='pretrained model path for generating instance dependent partial labels')

parser.add_argument('--data-dir', default='../data', type=str)

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18', 'resnet34'],
                    help='network architecture (only resnet18 used)')

parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')

parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-lr_decay_epochs', type=str, default='99,199,299',
                    help='where to decay lr, can be a list')

parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

parser.add_argument('--cosine', action='store_true', default=True,
                    help='use cosine lr schedule')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')


parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')


parser.add_argument('--cuda_VISIBLE_DEVICES', default='0', type=str, \
                        help='which gpu(s) can be used for distributed training')

parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')

parser.add_argument('--tau_proto', type=float, default=0.3,
                    help='temperature for prototype')

parser.add_argument('--alpha_mixup', type=float, default=8.0,
                    help='alpha for beta distribution')

parser.add_argument('--conf_th', type=float, default=0.5,
                    help='prototype update threshold')

parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')

parser.add_argument('--latent-dim', default=512, type=int,
                    help='latent embedding dimension')

parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')

parser.add_argument('--alpha_weight', default=1.0, type=float,
                    help='contrastive loss weight')

parser.add_argument('--pseudo_label_weight_range', default='0.95, 0.8', type=str,
                    help='pseudo target updating coefficient')

parser.add_argument('--pro_weight_range', default='0.9, 0.5', type=str,
                    help='prototype updating coefficient')

parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (q)')
parser.add_argument('--noisy_rate', default=0.1, type=float,
                    help='ambiguity level (q)')

parser.add_argument('--wp', default=5, type=int,
                    help='the number of epochs of warm-up training')
parser.add_argument('--k', default=5, type=int,
                    help='knn number')
parser.add_argument('--method', default='plrc', type=str, choices=['case0', 'plrc', 'alim-one-hot', 'alim-scale'],
                    help='methods')
parser.add_argument('--case', default='case2', type=str, help='cases')

parser.add_argument('--rho_range', default='0.7,1', type=str,
                    help='ratio of clean labels (rho)')
parser.add_argument('--rho_epoch', default=10, type=int,
                    help='ratio of clean labels (rho)')
parser.add_argument('--lamb', default=1.5, type=float,
                    help='reconstruction hyper-parameter')

parser.add_argument('--features', default='', type=str, choices=['blip2_feature_extractor', ''],
                    help='feature extraction model. default: original model')

parser.add_argument('--print2file', action='store_true', default=False,
                    help='whether to redirect std print to file')

parser.add_argument('--hierarchical', action='store_true', 
                    help='for CIFAR100-H training')

parser.add_argument('--save_result', action='store_true',
                    help='for CIFAR100-H training')

parser.add_argument('--prior', default=0, type=float, help='for ALIM')


args = parser.parse_args()
[args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(',')]

torch.set_printoptions(precision=2, sci_mode=False)

original_print = print
def print_with_time(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    original_print(current_time, end=" ")
    original_print(*args, **kwargs)

builtins.print = print_with_time

savedStdout = sys.stdout 
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    args.pseudo_label_weight_range = [float(item) for item in args.pseudo_label_weight_range.split(',')]
    args.pro_weight_range = [float(item) for item in args.pro_weight_range.split(',')]

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    model_path = 'h{h}_ds{ds}_{arch}_k{k}_ep{ep}_wp{wp}_rho_range{rr}_re{re}_pr{pr}_nr{nr}_sd{seed}_method{method}-fe_{features}-lamb{lamb}-lr{lr}-{time}'.format(
        ds=args.dataset,
        arch=args.arch,
        k=args.k,
        ep=args.epochs,
        wp=args.wp,
        rr=args.rho_range,
        re=args.rho_epoch,
        pr=args.partial_rate,
        nr=args.noisy_rate,
        seed=args.seed,
        method=args.method,
        time=time.strftime("%Y.%m.%d.%H:%M", time.localtime()),
        features=args.features,
        lamb=args.lamb,
        lr=args.lr,
        h=args.hierarchical
    )

    args.exp_dir = os.path.join(args.exp_dir, args.method, args.dataset, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    if args.print2file:
        print(f"The std output is saved in {os.path.join(args.exp_dir, 'output.log')}")
        sys.stdout = open(os.path.join(args.exp_dir, 'output.log'), 'w+')

    main_worker(args)

    if args.print2file:
        sys.stdout.close()
        sys.stdout = savedStdout  

    print('Done!')


def main_worker(args):
    if args.dataset == 'cifar10':
        args.num_class = 10
        train_loader, train_partialY_matrix, test_loader = load_noisy_cifar10(partial_rate=args.partial_rate, batch_size=args.batch_size, noisy_rate=args.noisy_rate, data_root='../data')
        args.dataset_size = train_partialY_matrix.shape[0]
    elif args.dataset == 'cifar100':
        args.num_class = 100
        train_loader, train_partialY_matrix, test_loader = load_noisy_cifar100(partial_rate=args.partial_rate, batch_size=args.batch_size, noisy_rate=args.noisy_rate, data_root='../data', hierarchical=args.hierarchical)
        args.dataset_size = train_partialY_matrix.shape[0]
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")


    print('\nAverage candidate num: {}\n'.format(train_partialY_matrix.sum(1).mean()))


    print("=> creating model '{}'\n".format(args.arch))

    model = PaPi(args, PaPiNet)

    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()
    
    loss_PaPi_func = PaPiLoss(predicted_score_cls=uniform_confidence, pseudo_label_weight=0.99)
    
    sim_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()

    if args.method == 'plrc':
        pllRecon = PLRC(partial_labels=train_partialY_matrix, partial_rate=args.partial_rate, noisy_rate=args.noisy_rate, rho_start=args.rho_start, rho_end=args.rho_end, start_epoch=args.wp, end_epoch=args.rho_epoch,
                     k=args.k, feature_extractor=args.features, dataset=args.dataset, seed=args.seed, lamb=args.lamb, args=args)
    else:
        pllRecon = None

    new_partial_labels = torch.Tensor(train_partialY_matrix).cuda()
    all_features = torch.empty((train_partialY_matrix.shape[0], args.low_dim))
    outputs_soft_labels = F.softmax(uniform_confidence + 1e-5, dim=1).cuda()
    best_acc = 0

    for epoch in range(args.start_epoch, args.epochs):

        is_best = False

        adjust_learning_rate(args, optimizer, epoch)
        
        loss_PaPi_func.set_alpha(epoch, args)
        if epoch > args.wp and args.method == 'plrc':
            pllRecon.update_rho(epoch=epoch)
            new_partial_labels = pllRecon(outputs_soft_labels=outputs_soft_labels, features=all_features)


        acc_train_cls, loss_cls_log, loss_PaPi_log, all_features, outputs_soft_labels = train(train_loader, model, loss_PaPi_func, optimizer, epoch, args, sim_criterion
                                                                         , partial_labels=new_partial_labels, all_features=all_features, outputs_soft_labels=outputs_soft_labels)
        
        loss_PaPi_func.set_pseudo_label_weight(epoch, args)
        model.set_prototype_update_weight(epoch, args)

        acc_test = test(model, test_loader)

        if acc_test > best_acc:
            best_acc = acc_test

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'.format(\
             epoch, acc_train_cls.avg, acc_test, best_acc, optimizer.param_groups[0]['lr']))

        print('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'.format(\
              epoch, acc_train_cls.avg, acc_test, best_acc, optimizer.param_groups[0]['lr']))


def train(train_loader, model, loss_PaPi_func, optimizer, epoch, args, sim_criterion, partial_labels=None, all_features=None, outputs_soft_labels=None):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@cls', ':2.2f')
    loss_PaPi_log = AverageMeter('Loss@PaPi', ':2.2f')

    if all_features is not None:
        features = all_features.detach().clone()
    if outputs_soft_labels is not None:
        soft_labels = outputs_soft_labels.detach().clone()
    model.train()

    end = time.time()
    for i, (images_1, images_2, labels, true_labels, index) in enumerate(train_loader):

        data_time.update(time.time() - end)
        
        X_1, X_2, Y = images_1.cuda(), images_2.cuda(), labels.cuda()
        Y_true = true_labels.long().detach().cuda()

        if partial_labels is not None:
            Y = partial_labels[index]

        Lambda = np.random.beta(args.alpha_mixup, args.alpha_mixup)
        idx_rp = torch.randperm(X_1.shape[0])
        X_1_rp = X_1[idx_rp]
        X_2_rp = X_2[idx_rp]
        # Y_rp = Y_selected[idx_rp]

        X_1_mix = Lambda * X_1 + (1 - Lambda) * X_1_rp
        X_2_mix = Lambda * X_2 + (1 - Lambda) * X_2_rp
        # Y_mix = Lambda * Y_selected + (1 - Lambda) * Y_rp

        cls_out_1, cls_out_2, logits_prot_1, logits_prot_2, logits_prot_1_mix, logits_prot_2_mix, feats_q = \
        model(img_q=X_1, img_k=X_2, img_q_mix=X_1_mix, img_k_mix=X_2_mix, partial_Y=Y, prior=args.prior)
        
        cls_loss_1, sim_loss_2, alpha_td = loss_PaPi_func(cls_out_1, cls_out_2, logits_prot_1, logits_prot_2
                                                          , logits_prot_1_mix, logits_prot_2_mix, idx_rp, Lambda, index, args, sim_criterion)

        loss = cls_loss_1 + alpha_td * sim_loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if all_features is not None:
            features[index] = feats_q.detach().cpu().clone()
        if outputs_soft_labels is not None:
            soft_labels[index] = F.softmax(cls_out_1.detach().clone(), dim=1)

        if epoch > args.wp and args.method.startswith('alim'):
            pseudo_label = F.softmax(cls_out_1.detach(), dim=1) * (Y + args.prior * (1 - Y))
            pseudo_label = pseudo_label / pseudo_label.sum(dim=1).repeat(pseudo_label.size(1), 1).transpose(0, 1)
            pseudo_label = pseudo_label.float().cuda().detach()
            if args.method == 'alim-one-hot':
                pseudo_label = F.one_hot(pseudo_label.max(dim=1)[1], pseudo_label.shape[-1]).float().cuda().detach()
            loss_PaPi_func.update_weight_byclsout1_alim(cls_pseudo_label=pseudo_label, batch_index=index)
        else:
            loss_PaPi_func.update_weight_byclsout1(cls_predicted_score=cls_out_1.detach(), batch_index=index, batch_partial_Y=Y)

        loss_cls_log.update(cls_loss_1.item())
        loss_PaPi_log.update(loss.item())

        acc = accuracy(cls_out_1.detach(), Y_true)
        acc_cls.update(acc[0].item())

        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 100 == 0) or ((i + 1) % len(train_loader) == 0):
            print('Epoch:[{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'A_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                'L_cls {Loss_cls.val:.4f} ({Loss_cls.avg:.4f})\t'
                'L_all {Loss_PaPi.val:.4f} ({Loss_PaPi.avg:.4f})\t'.format(
                    epoch, i + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, Acc_cls=acc_cls,
                    Loss_cls=loss_cls_log, Loss_PaPi=loss_PaPi_log
                )
            )
        
    return acc_cls, loss_cls_log, loss_PaPi_log, features, soft_labels



def test(model, test_loader):
    with torch.no_grad():

        print('\n=====> Evaluation...\n')
        model.eval()    

        acc_cnt = 0
        total_num = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            
            outputs, outputs_pro, _ = model(img_q=images, eval_only=True)
            total_num += outputs.size(0)
            acc_cnt += torch.eq(outputs.max(dim=1)[1], labels).sum().cpu()

        print('Top1 Accuracy is {:.2%}\n'.format(acc_cnt / total_num))

    return acc_cnt / total_num



if __name__ == '__main__':
    main()



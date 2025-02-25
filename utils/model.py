import numpy as np
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaPi(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        if args.dataset in ['miniImagenet', 'cub200', 'flower102', 'pet37', 'fgvc100']:
            pretrained = True
        else:
            pretrained = False
        
        self.proto_weight = args.proto_m

        self.encoder = base_encoder(name=args.arch, head='mlp', feat_dim=args.low_dim, num_class=args.num_class, pretrained=pretrained)

        self.register_buffer("prototypes", torch.zeros(args.num_class, args.low_dim))
    
    
    def set_prototype_update_weight(self, epoch, args):
        start = args.pro_weight_range[0]
        end = args.pro_weight_range[1]
        self.proto_weight = 1. * epoch / args.epochs * (end - start) + start


    def forward(self, img_q, img_k=None, img_q_mix=None, img_k_mix=None, partial_Y=None, eval_only=False, prior=0):

        output_q, q = self.encoder(img_q)
        
        if eval_only:
            prototypes_eval = self.prototypes.clone().detach()
            logits_prot_test = torch.mm(q, prototypes_eval.t())
            return output_q, logits_prot_test, q
        
        output_k, k = self.encoder(img_k)

        output_q_mix, q_mix = self.encoder(img_q_mix)
        output_k_mix, k_mix = self.encoder(img_k_mix)

        # predicetd_scores_q = torch.softmax(output_q, dim=1) * partial_Y
        # predicetd_scores_q_norm = predicetd_scores_q / predicetd_scores_q.sum(dim = 1).repeat(args.num_class, 1).transpose(0, 1)
        #
        # predicetd_scores_k = torch.softmax(output_k, dim=1) * partial_Y
        # predicetd_scores_k_norm = predicetd_scores_k / predicetd_scores_k.sum(dim = 1).repeat(args.num_class, 1).transpose(0, 1)
        
        pseudo_labels_q = torch.max(output_q * ( partial_Y + prior * (1 - partial_Y) ), dim=1)[1]
        pseudo_labels_k = torch.max(output_k * ( partial_Y + prior * (1 - partial_Y) ), dim=1)[1]
        
        prototypes = self.prototypes.clone().detach()
        
        logits_prot_q = torch.mm(q, prototypes.t())
        logits_prot_k = torch.mm(k, prototypes.t())

        logits_prot_q_mix = torch.mm(q_mix, prototypes.t())
        logits_prot_k_mix = torch.mm(k_mix, prototypes.t())

        for feat_q, label_q in zip(q.detach(), pseudo_labels_q):
            self.prototypes[label_q] = self.proto_weight * self.prototypes[label_q] + (1 - self.proto_weight) * feat_q
        
        for feat_k, label_k in zip(k.detach(), pseudo_labels_k):
            self.prototypes[label_k] = self.proto_weight * self.prototypes[label_k] + (1 - self.proto_weight) * feat_k
            
            
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        return output_q, output_k, logits_prot_q, logits_prot_k, logits_prot_q_mix, logits_prot_k_mix, q




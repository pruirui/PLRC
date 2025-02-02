import os
import time

import faiss
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
try:
    from lavis.models import load_model_and_preprocess
except:
    pass

from torch.utils.data import DataLoader

from utils.utils_dataset import get_dataset


class PLRC:
    '''
        PLL ReConstructor
    '''
    FEATURE_EXTRACTORS = {'blip2_feature_extractor':['pretrain', 'pretrain_vitL'], 'clip_feature_extractor':['ViT-B-32', 'ViT-B-16'], 'mocov2':['cifar10', 'cifar100']}
    FEATURE_ROOT_DIR = '../data'

    def __init__(self, partial_labels, partial_rate, noisy_rate, rho_start, rho_end, start_epoch, end_epoch, k=5, lamb=1.0, feature_extractor=None,  dataset=None, seed=1, true_labels=None, save_result=False, args=None):

        self.k = k
        self.feature_extractor = feature_extractor
        self.features = None

        if feature_extractor:
            start_time = time.time()
            assert feature_extractor in PLRC.FEATURE_EXTRACTORS
            dataset_path = os.path.join(PLRC.FEATURE_ROOT_DIR, dataset)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            file_path = os.path.join(dataset_path, f'{dataset}_{feature_extractor}_{seed}.pkl')
            if os.path.isfile(file_path):
                print(f'Loading features from {file_path}')
                self.features = torch.load(file_path)['features']
            else:
                if 'blip' in feature_extractor:
                    model, vis_processors, _ = load_model_and_preprocess(name=feature_extractor,
                                                                                      model_type=PLRC.FEATURE_EXTRACTORS[feature_extractor][0], is_eval=True,
                                                                                      device='cuda:0')
                    transform_feature = vis_processors['eval']
                elif 'clip' in feature_extractor:
                    model, vis_processors = load_model_and_preprocess(name=feature_extractor,
                                                                                      model_type=PLRC.FEATURE_EXTRACTORS[
                                                                                          feature_extractor][0], is_eval=True,
                                                                                      device='cuda:0')
                    transform_feature = vis_processors['eval']
                
                else:
                    raise ValueError(f'{feature_extractor} is not supported')
                    

                dataset = get_dataset(dataset_name=dataset, transform=transform_feature, root_dir='../data')
                dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
                with torch.no_grad():
                    model.eval()  # inference mode
                    features = []
                    for idx, res in enumerate(dataloader):
                        img = res[0]
                        img = img.cuda()
                        if 'blip' in feature_extractor:
                            sample = {"image": img, "text_input": None}
                            feature = model.extract_features(sample, mode="image").image_embeds[:, 0, :]
                        elif 'clip' in feature_extractor:
                            feature = model.encode_image(img).float()
                        elif 'mocov2' in feature_extractor:
                            feature = model(img)
                        features.append(feature.cpu())

                features = torch.vstack(features)
                self.features = features
                print(f'{feature_extractor} feature extraction time used: {time.time() - start_time}')
                torch.save({'features': self.features}, file_path)

            feat_dim = self.features.shape[-1]
            features = F.normalize(self.features, dim=1)
            faiss_index = faiss.IndexFlatIP(feat_dim)  
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
            faiss_index.add(features)  

            # 查询每个样本特征cosine距离最近的样本
            self.knn_distance, self.knn_index = faiss_index.search(features, self.k)  
        
        self.rho_start = rho_start
        self.rho_end = rho_end
        self.end_epoch = end_epoch
        self.start_epoch = start_epoch
        self.rho = rho_start
        self.lamb = lamb
        self.args = args

        self.partial_rate = partial_rate
        self.noisy_rate = noisy_rate
        self.num_class = partial_labels.shape[-1]
        self.partial_labels = partial_labels.detach().clone().cuda()
        self.partial_labels_new = self.partial_labels.detach().clone()
        self.outputs_soft_labels = torch.rand_like(self.partial_labels_new)
        self.mask = torch.ones((self.partial_labels_new.shape[0],)).bool()
        self.hc_idx = torch.arange(partial_labels.shape[0])
        self.non_hc_idx = torch.Tensor([])
        self.hard_idx = torch.Tensor([])

        self.save_result = save_result
        self.true_labels = true_labels
        self.normal_mask = None
        if true_labels is not None:
            self.true_labels = torch.Tensor(true_labels).long()
            self.normal_mask = self.partial_labels[range(len(self.partial_labels)), self.true_labels] == 1
        self.reliable_metric = []
        self.ls = []
        self.us = []
        self.normal_separation_acc = []
        self.noisy_separation_acc = []

        self.num_selected_samples = []
        self.mean_size_CLS = []
        self.normal_acc_reCLS = []
        self.noisy_acc_reCLS = []
        self.hard_acc_reCLS = []

        self.final_metric = []


    def update_rho(self, epoch, rho=None):
        if rho:
            self.rho = rho
        else:
            self.rho = self.rho_start + (self.rho_end - self.rho_start) * min(
                max((epoch - self.start_epoch) / (self.end_epoch - self.start_epoch), 0), 1)

    def __call__(self, outputs_soft_labels, features=None):


        device = outputs_soft_labels.device

        if self.features is None:
            assert features is not None
            features = F.normalize(features, dim=1)
            faiss_index = faiss.IndexFlatIP(features.shape[-1])  
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
            faiss_index.add(features)  

            knn_distance, knn_index = faiss_index.search(features, self.k)  
            knn_distance, knn_index = torch.from_numpy(knn_distance).to(device), torch.from_numpy(knn_index).to(device)
        else:
            features = F.normalize(self.features, dim=1)
            knn_distance, knn_index = torch.from_numpy(self.knn_distance).to(device), torch.from_numpy(self.knn_index).to(device)

        feat_dim = features.shape[-1]

        outputs_mask = outputs_soft_labels[knn_index]
        dis = knn_distance.reshape(knn_distance.shape[0], -1, 1)
        q_hat = F.softmax(torch.sum(outputs_mask * dis, dim=1), dim=1)

        revised_soft_labels = outputs_soft_labels * self.partial_labels
        revised_soft_labels = revised_soft_labels / revised_soft_labels.sum(dim=1, keepdim=True)

        # reliable_metric = torch.sum(q_hat * (torch.log(q_hat + 1e-6) - torch.log(revised_soft_labels + 1e-6)), dim=-1)
        # reliable_metric = torch.sum(revised_soft_labels * (torch.log(revised_soft_labels + 1e-6) - torch.log(q_hat + 1e-6)), dim=-1)
        reliable_metric = torch.sum(-revised_soft_labels * torch.log(q_hat + 1e-6), dim=1)



        idx_en = reliable_metric.sort()[1].cpu()


        selected_num = int(outputs_soft_labels.shape[0] * self.rho)
        hc_num = int(selected_num * (1 - self.noisy_rate))
        non_hc_num = selected_num - hc_num

        hc_idx = idx_en[:hc_num]
        non_hc_idx = idx_en[-non_hc_num:]

        # for observation
        self.hard_idx = idx_en[hc_num:-non_hc_num].cpu()
        self.hc_idx = hc_idx
        self.non_hc_idx = non_hc_idx
        self.outputs_soft_labels = outputs_soft_labels

        if self.save_result and self.normal_mask:
            self.reliable_metric.append([reliable_metric[self.normal_mask].cpu().numpy(), reliable_metric[torch.logical_not(self.normal_mask)].cpu().numpy()])
            self.ls.append(reliable_metric[hc_idx[-1]].cpu().item())
            self.us.append(reliable_metric[non_hc_idx[0]].cpu().item())



        new_partial_labels = self.partial_labels.detach().clone()
 
        faiss_index = faiss.IndexFlatIP(feat_dim)  
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
        selected_idx = torch.cat([hc_idx, non_hc_idx], dim=0)
        faiss_index.add(features[selected_idx])

        knn_distance, knn_index = faiss_index.search(features[selected_idx], self.k)
        knn_distance, knn_index = torch.from_numpy(knn_distance).to(device), torch.from_numpy(knn_index).to(
            device)

        non_hc_py = self.partial_labels[non_hc_idx]
        all_partial_idx = torch.where((1 - non_hc_py).sum(dim=1) == 0)
        f_non_hc_py = 1 - non_hc_py
        f_non_hc_py[all_partial_idx] = non_hc_py[all_partial_idx]

        revised_soft_labels[non_hc_idx] = outputs_soft_labels[non_hc_idx] * f_non_hc_py
        revised_soft_labels[non_hc_idx] = revised_soft_labels[non_hc_idx] / revised_soft_labels[non_hc_idx].sum(
            dim=1, keepdim=True)

        outputs_mask = revised_soft_labels[selected_idx][knn_index]
        dis = knn_distance.reshape(knn_distance.shape[0], -1, 1)
        q_hat = F.softmax(torch.sum(outputs_mask * dis, dim=1), dim=1)

        selected_py = self.partial_labels[selected_idx]
        selected_py[len(hc_idx):] = f_non_hc_py

        if self.args.case == 'case1':
            threshold = (q_hat * selected_py).max(dim=1)[0] / self.lamb
            threshold = (q_hat * selected_py > threshold.unsqueeze(-1)).sum(1)
        elif self.args.case == 'case2':
            threshold = (1 - q_hat.max(dim=1)[0]) * self.partial_rate * self.num_class + self.lamb
        elif self.args.case == 'case3':
            threshold = 1 / self.lamb
            threshold = (q_hat * selected_py > threshold).sum(1)
        else:
            raise ValueError(f"No case! {self.args['case']}")

        threshold = torch.clamp(threshold, torch.ones_like(threshold), self.partial_labels[selected_idx].sum(1)).long()

        selected_py_new = torch.zeros_like(selected_py)
        vals, idxs = torch.sort(q_hat * selected_py, dim=1, descending=True)
        for i in range(len(vals)):
            selected_py_new[i, idxs[i, :threshold[i]]] = 1

        new_partial_labels[selected_idx] = selected_py_new

        # for hard sample
        hard_pll = self.partial_labels[self.hard_idx]
        hard_py_new = hard_pll.detach().clone()
        max_idx = torch.max(outputs_soft_labels[self.hard_idx] * (1 - hard_pll), dim=1)[1]
        min_idx = torch.min(outputs_soft_labels[self.hard_idx] + (1 - hard_pll), dim=1)[1]
        hard_py_new[range(len(hard_pll)), max_idx] = 1
        hard_py_new[range(len(hard_pll)), min_idx] = 0
        new_partial_labels[self.hard_idx] = hard_py_new


        self.partial_labels_new = new_partial_labels

        return self.partial_labels_new

    def show_details(self, true_labels, save_path=None):
        true_labels = torch.Tensor(true_labels).long()
        'begin for observation'
        non_hc_py = self.partial_labels[self.non_hc_idx]
        hard_py = self.partial_labels[self.hard_idx]
        hc_py = self.partial_labels[self.hc_idx]
        non_hc_py_new = self.partial_labels_new[self.non_hc_idx]
        hard_py_new = self.partial_labels_new[self.hard_idx]
        hc_py_new = self.partial_labels_new[self.hc_idx]

        hc_gt = true_labels[self.hc_idx].long()
        hc_gt_in_pll_ratio = ((hc_py[range(len(hc_py)), hc_gt] == 1).cpu().sum() / len(hc_py)).item()
        hc_hit_rate = ((hc_py_new[range(len(self.hc_idx)), hc_gt] == 1).cpu().sum() / len(hc_py_new)).item()
        hc_confidence = self.outputs_soft_labels[self.hc_idx].cpu().max(dim=1)[0].mean().item()
        hc_acc = ((self.outputs_soft_labels[self.hc_idx].cpu().max(dim=1)[1] == hc_gt).sum() / len(self.hc_idx)).item()

        non_hc_gt = true_labels[self.non_hc_idx].long()
        non_hc_gt_in_pll_ratio = (
                (non_hc_py[range(len(non_hc_py)), non_hc_gt] == 1).cpu().sum() / len(non_hc_py)).item()
        non_hc_hit_rate = (
                (non_hc_py_new[range(len(non_hc_py_new)), non_hc_gt] == 1).cpu().sum() / len(non_hc_py_new)).item()
        non_hc_confidence = self.outputs_soft_labels[self.non_hc_idx].cpu().max(dim=1)[0].mean().item()
        non_hc_acc = ((self.outputs_soft_labels[self.non_hc_idx].cpu().max(dim=1)[1] == non_hc_gt).sum() / len(
            self.non_hc_idx)).item()

        if len(self.hard_idx) > 0:
            hard_gt = true_labels[self.hard_idx].long()
            hard_partial_cnt = (self.partial_labels[self.hard_idx][range(len(self.hard_idx)), hard_gt] == 1).cpu()
            hard_gt_in_pll_ratio = (hard_partial_cnt.sum() / len(hard_partial_cnt)).item()
            hard_hit_rate = ((hard_py_new[range(len(hard_py_new)), hard_gt] == 1).cpu().sum() / len(hard_py_new)).item()

            hard_confidence = self.outputs_soft_labels[self.hard_idx].cpu().max(dim=1)[0].mean().item()
            hard_acc_mask = self.outputs_soft_labels[self.hard_idx].cpu().max(dim=1)[1] == hard_gt
            bingo_num = hard_acc_mask.sum()
            bingo_idx = torch.where(hard_acc_mask)
            hard_bingo_in_pll_ratio = ((self.partial_labels[self.hard_idx][bingo_idx].cpu()[
                                            range(bingo_num), hard_gt[bingo_idx]] == 1).sum() / bingo_num).item()
            hard_acc = (bingo_num / len(self.hard_idx)).item()

            hard_new_pl_avg = hard_py_new.sum(dim=1).mean().item()
        else:
            hard_hit_rate = 0
            hard_confidence = 0
            hard_gt_in_pll_ratio = 0
            hard_acc = 0
            hard_bingo_in_pll_ratio = 0
            hard_new_pl_avg = 0

        hc_new_pl_avg = hc_py_new.sum(dim=1).mean().item()
        non_hc_new_pl_avg = non_hc_py_new.sum(dim=1).mean().item()


        if save_path is not None and self.save_result:
            cur_normal_mask = self.partial_labels_new[range(len(true_labels)), true_labels] == 1
            epsi = torch.sum(torch.logical_not(cur_normal_mask)) / len(true_labels)
            mean_normal_size_cls = self.partial_labels_new[cur_normal_mask].sum() / cur_normal_mask.sum()
            self.final_metric.append(1 - (1 - epsi.cpu().item()) / mean_normal_size_cls.cpu().item())

            self.normal_separation_acc.append(hc_gt_in_pll_ratio)
            self.noisy_separation_acc.append(1 - non_hc_gt_in_pll_ratio)

            self.num_selected_samples.append(len(self.non_hc_idx) + len(self.hc_idx))
            self.mean_size_CLS.append((self.partial_labels_new.sum() / len(true_labels)).cpu().item())
            self.normal_acc_reCLS.append(hc_hit_rate / hc_gt_in_pll_ratio)
            self.noisy_acc_reCLS.append(non_hc_hit_rate / (1 - non_hc_gt_in_pll_ratio))
            self.hard_acc_reCLS.append(hard_hit_rate)

            torch.save({'reliable_metric': self.reliable_metric, 'ls':self.ls, 'us':self.us,
                        'num_selected_samples':self.num_selected_samples, 'normal_separation_acc':self.normal_separation_acc, 'noisy_separation_acc':self.noisy_separation_acc,
                        'mean_size_CLS':self.mean_size_CLS, 'normal_acc_reCLS':self.normal_acc_reCLS, 'noisy_acc_reCLS':self.noisy_acc_reCLS, 'hard_acc_reCLS':self.hard_acc_reCLS,
                        'final_metric':self.final_metric}, save_path)

        return f'''
rho: {self.rho}
Selecting candidate label for normal sample. The avg. pll:{hc_new_pl_avg}
Selecting candidate label for hard sample. The avg. pll:{hard_new_pl_avg}
Selecting candidate label for noisy sample. The avg. pll:{non_hc_new_pl_avg}
partial label hit rate hc(revised/original): {hc_hit_rate:.2%}/{hc_gt_in_pll_ratio:.2%}, hard(revised/original): {hard_hit_rate:.2%}/{hard_gt_in_pll_ratio:.2%}, non_hc(revised/original): {non_hc_hit_rate:.2%}/{non_hc_gt_in_pll_ratio:.2%}
prediction confidence hc: {hc_confidence:.4f}, hard: {hard_confidence:.4f}, non_hc: {non_hc_confidence:.4f}
prediction acc hc: {hc_acc:.2%}, hard: {hard_acc:.2%}, non_hc: {non_hc_acc:.2%}
hard bingo in candidate set ratio: {hard_bingo_in_pll_ratio:.2%}'''

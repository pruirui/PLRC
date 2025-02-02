import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.randaugment import RandomAugment
import PIL.Image


def get_data(data_root, base_dir, num_label=2):
    import json
    from collections import defaultdict
    import torch
    import random
    save_path = os.path.join(data_root, base_dir, f'{base_dir}_selected_num_{num_label}.pt')
    if os.path.exists(save_path):
        res = torch.load(save_path)
        train_data, train_gts, train_pys, test_data, test_gts = res['train_data'], res['train_gts'], res['train_pys'], res['test_data'], res['test_gts']
        nr = res['noise_rate']
        cls = res['mean_cls']
        labels = res['labels']
        print('loading data')
        print(
            f'noise rate:{nr}, mean cls:{cls}, train sample num:{len(train_data)}, test sample num:{len(test_data)}, label length:{len(labels)}')
        print('labels:', labels)
        return train_data, train_gts, train_pys, test_data, test_gts

    json_path = os.path.join(data_root, base_dir, 'annotations.json')
    # 读取文件内容到字符串中
    with open(json_path, 'r', encoding='utf-8') as file:
        json_str = file.read()

    annotations = json.loads(json_str)[0]['annotations']
    imgs = defaultdict(dict)
    labels = set()

    for record in annotations:
        file_name = record['image_path']
        class_label = record['class_label']
        labels.add(class_label)
        if class_label not in imgs[file_name]:
            imgs[file_name][class_label] = 1
        else:
            imgs[file_name][class_label] += 1
    labels = sorted(list(labels))
    label2idx = {v:i for i, v in enumerate(labels)}
    partial_y = torch.zeros((len(imgs), len(labels)))
    gts = []
    data = []
    total_cnt = 0
    for i, (file_path, img) in enumerate(imgs.items()):
        label_list = []
        max_val = -1
        gt = random.randint(0, len(labels))
        for label, count in img.items():
            label_list.extend([label] * count)
            if count > max_val:
                gt = label2idx[label]
                max_val = count

        total_cnt += sum(img.values())
        data.append(os.path.join(data_root, file_path))
        gts.append(gt)
        py_label = [label2idx[v] for v in random.sample(label_list, min(len(label_list), num_label))]
        partial_y[i, py_label] = 1

    train_idx = torch.randperm(len(data))[:int(len(data) * 0.8)]
    train_data = [data[idx] for idx in train_idx.tolist()]
    train_gts = [gts[idx] for idx in train_idx.tolist()]
    train_pys = partial_y[train_idx]

    test_idx = torch.randperm(len(data))[int(len(data) * 0.8):]
    test_data = [data[idx] for idx in test_idx.tolist()]
    test_gts = [gts[idx] for idx in test_idx.tolist()]
    test_pys = partial_y[test_idx]

    nr = torch.sum(train_pys[range(len(train_data)), train_gts] == 0).item() / len(train_data)
    cls = train_pys.sum().item() / len(train_data)
    print(f'noise rate:{nr}, mean cls:{cls}, train sample num:{len(train_data)}, test sample num:{len(test_data)}, label length:{partial_y.shape[-1]}')
    print('avg cnt:', total_cnt / len(data))
    print('labels:',labels)
    print('saving dataset!')
    torch.save({'train_data': train_data,
                'train_gts': train_gts,
                'train_pys': train_pys,
                'test_data': test_data,
                'test_gts': test_gts,
                'test_pys': test_pys,
                'noise_rate': nr,
                'mean_cls': cls,
                'labels': labels
                }, os.path.join(data_root, base_dir, f'{base_dir}_selected_num_{num_label}.pt'))

    return train_data, train_gts, train_pys, test_data, test_gts

def load_treeversity_dataset(data_root='../data', num_selected=3, train=True, transform=None):
    base_dir = 'Treeversity#6'
    train_data, train_gts, train_pys, test_data, test_gts = get_data(data_root, base_dir, num_label=num_selected)
    train_gts = torch.Tensor(train_gts).long()
    test_gts = torch.Tensor(test_gts).long()
    if train:
        return Treeviersity(image_files=train_data, given_partial_label_matrix=train_pys, true_labels=train_gts, test_transform=transform)
    else:
        return Treeviersity(image_files=test_data, given_partial_label_matrix=None, true_labels=test_gts, test_transform=transform)

def load_treeversity(batch_size, data_root='../data', num_selected=2):
    base_dir = 'Treeversity#6'

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    train_data, train_gts, train_pys, test_data, test_gts = get_data(data_root, base_dir, num_label=num_selected)
    train_gts = torch.Tensor(train_gts).long()
    test_gts = torch.Tensor(test_gts).long()


    test_dataset = Treeviersity(image_files=test_data, given_partial_label_matrix=None, true_labels=test_gts, test_transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size * 4, \
        shuffle=False, num_workers=8, pin_memory=False
    )

    partial_training_dataset = Treeviersity(image_files=train_data, given_partial_label_matrix=train_pys, true_labels=train_gts)

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return partial_training_dataloader, train_pys, test_loader

class Treeviersity(Dataset):
    def __init__(self, image_files, given_partial_label_matrix, true_labels, test_transform=None):

        self.image_files = image_files
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        self.test_transform = test_transform

        self.weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):

        image_file = self.image_files[index]
        image = PIL.Image.open(image_file).convert("RGB")
        if self.test_transform:
            return self.test_transform(image), self.true_labels[index]

        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        return each_image_w, each_image_s, each_label, each_true_label, index


def generate_uniform_noisy_candidate_labels(train_labels, partial_rate=0.1, noisy_rate=0):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    # inject label noise if noisy_rate > 0
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    for j in range(n):  # for each instance
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)

    if noisy_rate == 0:
        partialY[torch.arange(n), train_labels] = 1.0
        # if supervised, reset the true label to be one.
        print('Reset true labels')

    print("Finish Generating Candidate Label Sets!\n")
    return partialY
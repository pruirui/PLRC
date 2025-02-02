import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.randaugment import RandomAugment
from torchvision.datasets import Flowers102
import PIL.Image


def load_flower102(partial_rate, batch_size, noisy_rate=0, data_root='../data'):
    base_dir = 'FLOWER102'
    data_root = os.path.join(data_root, base_dir)
    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    original_train = Flowers102(root=data_root, split='train', download=True)
    ori_labels = torch.Tensor(original_train._labels).long()

    test_dataset = Flowers102(root=data_root, split='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size * 4, \
        shuffle=False, num_workers=8, pin_memory=False
    )
    partialY_matrix = generate_uniform_noisy_candidate_labels(train_labels=ori_labels, partial_rate=partial_rate, noisy_rate=noisy_rate)


    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1

    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print(f'noisy partial label data load ! noise rate: {1 - torch.sum(partialY_matrix * temp) / partialY_matrix.shape[0]}')

    print('Average candidate num: ', partialY_matrix.sum(1).mean())

    partial_training_dataset = FLOWER102_Partialize(original_train._image_files, partialY_matrix.float(), ori_labels)

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return partial_training_dataloader, partialY_matrix, test_loader

class FLOWER102_Partialize(Dataset):
    def __init__(self, image_files, given_partial_label_matrix, true_labels):

        self.image_files = image_files
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels


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
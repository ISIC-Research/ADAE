import os

import albumentations
import cv2
import geffnet
import numpy as np
import pandas as pd
from pretrainedmodels import se_resnext101_32x4d
from resnest.torch import resnest101
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


sigmoid = nn.Sigmoid()

def get_transforms(image_size):

    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()])


class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()
        
    def extract(self, x):
        return self.enet(x)

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
        
    def forward_and_maps(self, x, x_meta=None, mmult=None):
        x = self.enet.conv_stem(x)
        x = self.enet.bn1(x)
        x = self.enet.act1(x)
        x = self.enet.blocks(x)
        x = self.enet.conv_head(x)
        x = self.enet.bn2(x)
        maps = self.enet.act2(x)
        if mmult is not None:
            maps = torch.mul(mmult, maps)
        emb = self.enet.global_pool(maps).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((emb, x_meta), dim=1)
        else:
            x = emb
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        mapsh = maps.shape
        maps = maps.permute(0, 2, 3, 1).reshape((mapsh[0]*mapsh[2]*mapsh[3],mapsh[1]))
        maps = torch.matmul(maps, self.myfc.weight[:,:mapsh[1]].T)
        maps = maps.reshape((mapsh[0], mapsh[2], mapsh[3], maps.shape[1])).permute(0, 3, 1, 2)
        return out, emb, maps


class Resnest_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Resnest_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()
        
    def extract(self, x):
        return self.enet(x)

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
        
    def forward_and_maps(self, x, x_meta=None, mmult=None):
        x = self.enet.conv1(x)
        x = self.enet.bn1(x)
        x = self.enet.relu(x)
        x = self.enet.maxpool(x)
        x = self.enet.layer1(x)
        x = self.enet.layer2(x)
        x = self.enet.layer3(x)
        maps = self.enet.layer4(x)
        if mmult is not None:
            maps = torch.mul(mmult, maps)
        emb = self.enet.avgpool(maps).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((emb, x_meta), dim=1)
        else:
            x = emb
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        mapsh = maps.shape
        maps = maps.permute(0, 2, 3, 1).reshape((mapsh[0]*mapsh[2]*mapsh[3],mapsh[1]))
        maps = torch.matmul(maps, self.myfc.weight[:,:mapsh[1]].T)
        maps = maps.reshape((mapsh[0], mapsh[2], mapsh[3], maps.shape[1])).permute(0, 3, 1, 2)
        return out, emb, maps


class Seresnext_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Seresnext_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.last_linear = nn.Identity()
        
    def extract(self, x):
        return self.enet(x)

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
        
    def forward_and_maps(self, x, x_meta=None, mmult=None):
        x = self.enet.layer0(x)
        x = self.enet.layer1(x)
        x = self.enet.layer2(x)
        x = self.enet.layer3(x)
        maps = self.enet.layer4(x)
        if mmult is not None:
            maps = torch.mul(mmult, maps)
        emb = self.enet.avg_pool(maps).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((emb, x_meta), dim=1)
        else:
            x = emb
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        mapsh = maps.shape
        maps = maps.permute(0, 2, 3, 1).reshape((mapsh[0]*mapsh[2]*mapsh[3],mapsh[1]))
        maps = torch.matmul(maps, self.myfc.weight[:,:mapsh[1]].T)
        maps = maps.reshape((mapsh[0], mapsh[2], mapsh[3], maps.shape[1])).permute(0, 3, 1, 2)
        return out, emb, maps


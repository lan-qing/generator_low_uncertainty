import numpy as np
import torch
import torch.nn as nn

from utils import *
from models.mlp_vi import MLP
from models.lenet_vi import Lenet
from attacks import pgd_attack_reverse, pgd_reverse_on_uncertainty
from visualization import to_img


class NetWrapper():
    def __init__(self):
        cprint('c', '\nNet:')
        self.model = None

    def fit(self, train_loader):
        raise NotImplementedError

    def predict(self, test_loader):
        raise NotImplementedError

    def validate(self, val_loader):
        raise NotImplementedError

    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['state_dict'])


class MLPWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False):
        super(MLPWrapper).__init__()
        self.N = N
        self.model = MLP(widths=[784, 1200, 1200, 10], N=N)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0.0, epoch=None, adv=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double,
                           adv=adv)

        return loss, prec

    def validate(self, val_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double)
        return loss, prec

    def generate_from_noise(self):
        self.model.eval()
        noise_image = torch.rand([1, 784])
        to_img(torch.reshape(noise_image, [28, 28]), "tmp_ori.png")
        target = pgd_attack_reverse(self.model, noise_image, torch.tensor([2]))
        return target

    def generate_from_noise_low_uncertainty(self):
        self.model.eval()
        noise_image = torch.rand([1, 784])
        to_img(torch.reshape(noise_image, [28, 28]), "tmp_ori.png")
        target = pgd_reverse_on_uncertainty(self.model, noise_image, torch.tensor([2]))
        return target

class LenetWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False):
        super(LenetWrapper).__init__()
        self.N = N
        self.model = Lenet(N=N)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0.0, epoch=None, adv=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double,
                           adv=adv)

        return loss, prec

    def validate(self, val_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double)
        return loss, prec

    def generate_from_noise(self):
        self.model.eval()
        noise_image = torch.rand([1, 784])
        to_img(torch.reshape(noise_image, [28, 28]), "tmp_ori.png")
        target = pgd_attack_reverse(self.model, noise_image, torch.tensor([0]))
        return target

    def generate_from_noise_low_uncertainty(self):
        self.model.eval()
        noise_image = torch.rand([1, 784])
        to_img(torch.reshape(noise_image, [28, 28]), "tmp_ori.png")
        target = pgd_reverse_on_uncertainty(self.model, noise_image, torch.tensor([8]))
        return target
import numpy as np
import torch
import torch.nn as nn

from utils import *
from models.mlp_vi import MLP


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
    def __init__(self, half=True, cuda=True, double=False):
        super(MLPWrapper).__init__()
        self.model = MLP()
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.1, weight_decay=0.0, epoch=None, adv=None):
        optimizer = torch.optim.SGD(self.model.parameters(), lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        if self.double:
            criterion.double()

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, criterion, optimizer, epoch, half=self.half, double=self.double,
                           adv=adv)

        return loss, prec

    def predict(self, test_loader):
        pass

    def validate(self, val_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double)
        return loss, prec

    def generate_adv_data(self, attack, data_loader, print_freq=5):
        self.model.eval()
        batch_data = []
        batch_time = AverageMeter()
        end = time.time()
        for i, (input_data, target) in enumerate(data_loader):
            if self.half:
                input_data = input_data.half()
            if self.double:
                input_data = input_data.double()
            image = attack(self.model, input_data, target, double=self.double)
            batch_data.append(image.cpu())
            if i % print_freq == 0:
                batch_time.update(time.time() - end)
                end = time.time()
                print('Adv Gen: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time))
        return np.concatenate(batch_data, axis=0)

    def generate_poison_data_from_data(self, target_img, base_img, max_iter=200, target_feat=None, coeff_sim_inp=0.25,
                             max_difference=0.05):
        if target_img is not None:
            assert target_feat is None
        else:
            assert target_feat is not None

        batch_size = None
        self.model.eval()

        if target_img is not None:
            if base_img.shape == target_img.shape:
                batch_size = 0
            elif base_img.shape[1:] == target_img.shape:
                batch_size = base_img.shape[0]
            else:
                print("Sizes not match error")
                exit(-1)

            target_img = torch.unsqueeze(target_img, 0)
            if not batch_size:
                base_img = torch.unsqueeze(base_img, 0)
                batch_size = 1

            with torch.no_grad():
                target_feat = self.resnet(self.norm_layer(target_img), penultimate=True)

        else:
            if len(target_feat.shape == 2):
                batch_size = len(target_feat)
            elif len(target_feat.shape == 1):
                batch_size = 1
                target_feat = target_feat.unsqueeze(target_feat, 0)
            else:
                print("Sizes not match error")
                exit(-1)
        assert batch_size
        x = base_img.clone()

        learning_rate = 1

        for it in range(max_iter):
            x = x.clone()
            x.requires_grad = True
            x_feat = self.resnet(self.norm_layer(x), penultimate=True)
            loss = ((target_feat - x_feat) ** 2).mean() * batch_size
            self.model.zero_grad()
            loss.backward()
            optimizer = torch.optim.SGD([x], learning_rate)
            optimizer.step()
            with torch.no_grad():
                x = (x + learning_rate * coeff_sim_inp * base_img) / (1 + coeff_sim_inp * learning_rate)
                x = torch.clip(x, 0.0, 1.0)
        dif = x - base_img
        dif = torch.clamp(dif, -max_difference, max_difference)
        x = dif + base_img
        return x
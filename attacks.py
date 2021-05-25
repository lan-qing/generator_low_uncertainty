import torch
import torch.nn as nn
import numpy as np


def pgd_attack_reverse(model, images, labels, eps=1.0, alpha=0.01, iters=1000, half=False, double=False):
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs, kl = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()
        print(torch.argmax(outputs))
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def pgd_reverse_on_uncertainty(model, images, labels, eps=1.0, alpha=0.01, iters=1000, half=False, double=False, samplings=5):
    images = images.cuda()

    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        if i % 4 != 0:
            outputs, kl = model(images)

            model.zero_grad()
            cost = loss(outputs / 100000, labels)
            cost.backward()
            if i % 10 == 0:
                print(i, cost, outputs)
                print(torch.argmax(outputs))
            adv_images = images - alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        else:
            Hs = []
            for j in range(samplings):
                outputs, kl = model(images)
                Hs.append(outputs.reshape(-1) / 100)
            Hs = torch.stack(Hs)
            Ha = (Hs.std(dim=1) ** 2).mean()
            He = (Hs.mean(dim=1).std()) ** 2
            model.zero_grad()
            cost = Ha + He
            cost.backward()
            if i % 10 == 0:
                print(i, cost)
                print(torch.argmax(outputs))
            adv_images = images - alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images

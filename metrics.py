import torch
import torch.nn.functional as F

from sklearn.metrics import auc

from util import normalize
from util import norm

def deletion3d(model, input, cam, target=None, nsteps=100, return_maskedinput=False):
    masked_input = []
    probabilities = []
    _input = input.clone().detach()
    idx = cam.flatten().sort(descending=True)[1]
    for step in torch.linspace(0, len(idx), nsteps).long():
        _input = _input.transpose(1,0).flatten(1)
        _input[:, idx[:step]] = 0
        _input = _input.view(input.transpose(1,0).shape).transpose(1,0)
        _output = model(normalize(_input).permute(1,0,2,3).cuda().unsqueeze(0))
        probabilities.append(torch.softmax(_output, 1).detach().cpu())
        if return_maskedinput:
            masked_input.append(_input.clone().detach().cpu())
    probabilities = torch.cat(probabilities)
    if target is not None:
        if return_maskedinput:
            return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target]), torch.cat(masked_input)
        return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target])
 
    if return_maskedinput:
        return probabilities, torch.cat(masked_input)
    return probabilities

def insertion3d(model, input, cam, target=None, nsteps=100, return_maskedinput=False, factor=16):
    masked_input = []
    probabilities = []
    _input = input.clone().detach()
    idx = cam.flatten().sort(descending=True)[1]
    _blurred = F.interpolate(
        F.interpolate(
            input.clone().detach().transpose(0,1).unsqueeze(0), 
            scale_factor=1/factor, 
            mode='trilinear', 
            align_corners=True,
            recompute_scale_factor=True), 
        size=input.transpose(0,1).shape[-3:], 
        mode='trilinear', 
        align_corners=True).transpose(1,2).squeeze()
    for step in torch.linspace(0, len(idx), nsteps).long():
        _blurred.transpose(1,0).flatten(1)[:, idx[:step]] = _input.transpose(1,0).flatten(1)[:,idx[:step]]
        _output = model(normalize(_blurred).permute(1,0,2,3).cuda().unsqueeze(0))
        probabilities.append(torch.softmax(_output, 1).detach().cpu())
        if return_maskedinput:
            masked_input.append(_input.clone().detach().cpu())
    probabilities = torch.cat(probabilities)
    if target is not None:
        if return_maskedinput:
            return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target]), torch.cat(masked_input)
        return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target])
 
    if return_maskedinput:
        return probabilities, torch.cat(masked_input)
    return probabilities


def deletionPAN(model, input, cam, target=None, nsteps=100, return_maskedinput=False):
    masked_input = []
    probabilities = []
    _input = input.clone().detach()
    idx = cam.repeat_interleave(12,0).unsqueeze(0).flatten().sort(descending=True)[1]
    for step in torch.linspace(0, len(idx), nsteps).long():
        _input = _input.flatten()
        _input[idx[:step]] = 0
        _input = _input.view(input.shape)

        _output = model(norm(_input).permute(1,0,2,3).cuda().unsqueeze(0))
        probabilities.append(torch.softmax(_output, 1).detach().cpu())

        if return_maskedinput:
            masked_input.append(_input.clone().detach().cpu())
    probabilities = torch.cat(probabilities)
    probabilities[probabilities.isnan()] = 0
    if target is not None:
        if return_maskedinput:
            return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target]), torch.cat(masked_input)
        return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target])
 
    if return_maskedinput:
        return probabilities, torch.cat(masked_input)
    return probabilities


def insertionPAN(model, input, cam, target=None, nsteps=100, return_maskedinput=False, factor=16):
    masked_input = []
    probabilities = []
    _input = input.clone().detach()
    idx = cam.repeat_interleave(12,0).unsqueeze(0).flatten().sort(descending=True)[1]
    
    _blurred = F.interpolate(
        F.interpolate(input.clone().detach(), scale_factor=1/factor, mode='bilinear', align_corners=True,recompute_scale_factor=True),
        size=input.shape[-2:], 
        mode='bilinear', 
        align_corners=True)

    for step in torch.linspace(0, len(idx), nsteps).long():
        _blurred = _blurred.flatten()
        _blurred[idx[:step]] = _input.flatten()[idx[:step]]
        _blurred = _blurred.view(input.shape)

        _output = model(norm(_blurred).cuda().unsqueeze(0))
        probabilities.append(torch.softmax(_output, 1).detach().cpu())

        if return_maskedinput:
            masked_input.append(_blurred.clone().detach().cpu())
    probabilities = torch.cat(probabilities)
    probabilities[probabilities.isnan()] = 0
    if target is not None:
        if return_maskedinput:
            return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target]), torch.cat(masked_input)
        return probabilities[:, target], auc(torch.linspace(0, 1, steps=nsteps), probabilities[:,target])
 
    if return_maskedinput:
        return probabilities, torch.cat(masked_input)
    return probabilities
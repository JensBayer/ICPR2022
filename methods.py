from functools import partial

import torch
import torch.nn.functional as F

import numpy as np
from scipy.spatial.distance import cdist
from skimage.transform import resize

from util import normalize, norm

def gradcam3d(model, layer, input, target=None):
    features = []
    grad = []

    def fwd_hook(module, in_features, out_features, fout=None):
        fout += in_features

    def bck_hook(module, in_grad, out_grad, fout=None):
        fout += in_grad

    fwd = layer.register_forward_hook(partial(fwd_hook, fout=features))
    bck = layer.register_full_backward_hook(partial(bck_hook, fout=grad))
    
    output = model(normalize(input).permute(1,0,2,3).cuda().unsqueeze(0))
    if target is not None:
        output[0, target].backward()
    else:
        output.max().backward()
    fwd.remove()
    bck.remove()
    features = features[0]
    grad = grad[0]

    α = grad.sum((2,3,4))
    cam = (torch.nn.functional.interpolate(torch.relu(features*α.view(1,-1,1,1,1)), [len(input), *(input.shape[2:])], mode='trilinear', align_corners=True)).sum(1)
    
    return cam

def rise3d(model, input, target=None, gpu_batch=24, N=1000, s=8, p1=0.1):
    B, C, T, H, W = input.size()
    input_size = (T, H, W)
    cell_size = torch.ceil(torch.tensor(input_size) / s).long()
    up_size = ((s + 1) * cell_size).tolist()

    grid = (torch.rand(N, s//4, s, s) < p1).float()

    masks = torch.zeros((N, *input_size))
    shifts = []
    for i in range(N):
            t = torch.randint(0, cell_size[0], [1])
            x = torch.randint(0, cell_size[1], [1])
            y = torch.randint(0, cell_size[2], [1])
            shifts.append((t, x, y))
            masks[i, :, :] = F.interpolate(grid[[i]].unsqueeze(0).cuda(), up_size, mode='trilinear', align_corners=True).squeeze()[t:t + input_size[0], x:x + input_size[1], y:y + input_size[2]].cpu()
    masks = masks.reshape(-1, 1, *input_size)


    with torch.no_grad():
        p = []
        x = input.cuda()
        for i in range(0, N, gpu_batch):
            batch = masks[i:min(i + gpu_batch, N)].cuda() * x
            batch = normalize(batch.transpose(1,2).flatten(0,1)).view(batch.transpose(1,2).shape).transpose(1,2)
            
            p.append(model(batch).cpu())
            del batch

        p = torch.cat(p)
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), masks.view(N, T * H * W))
        sal = sal.view((CL, T, H, W))
        sal = sal / N / p1
        if target is not None:
            sal = sal[target]

        return sal

#see https://github.com/satyamahesh84/SIDU_XAI_CODE.git
@torch.no_grad()
def sidu3d(model, layer, input, target=None, batch_size=64, τ=0.5):    
    def generate_masks_conv_output(input_size, last_conv_output, s=8, τ=0.5):
        up_size = (torch.tensor(input_size) / s).ceil() * s
        masks = F.interpolate((last_conv_output > τ).float(), size=input_size, mode='trilinear', align_corners=True).permute(0,2,3,4,1).squeeze()
        return mask
    
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))  

    def sim_differences(pred_org, preds):
        diff = abs(pred_org-preds)
        weights= kernel(diff,0.25)
        return weights, diff

    def uniqness_measure(masks_predictions):
        sum_all_cdist =(torch.cdist(masks_predictions, masks_predictions)).sum(axis=1)
        sum_all_cdist = norm(sum_all_cdist)
        return sum_all_cdist
    
    def explain_SIDU(model, inp, N, p1, masks, input_size):
        preds = []
        pred_org = model(normalize(inp).transpose(0,1).unsqueeze(0).cuda()).detach().cpu().numpy()

        with torch.no_grad():
            masks = masks.unsqueeze(2).float().cuda()
            inp = inp.unsqueeze(0).cuda()
            for i in range(0, N, batch_size):
                batch = masks[i:min(i+batch_size, N)] * inp
                batch = normalize(batch.flatten(0,1)).view(batch.shape)
                preds.append(model(batch.transpose(1,2)))

            preds = torch.cat(preds).detach().cpu().numpy()
            weights, diff = sim_differences(pred_org, preds)
            interactions = uniqness_measure(torch.tensor(preds))
            new_interactions = interactions.reshape(-1, 1)
            diff_interactions = np.multiply(weights, new_interactions)
            sal = diff_interactions.numpy().T.dot(masks.squeeze().cpu().numpy().reshape(N, -1)).reshape(-1, *input_size)
            sal = sal / N / p1
        
        return sal, weights, new_interactions, diff_interactions, pred_org
    
    
    features = []

    
    model.eval()
    
    def fwd_hook(module, in_features, out_features, fout=None):
        fout += in_features

    fwd = layer.register_forward_hook(partial(fwd_hook, fout=features))
    output = model(norm(input.transpose(0,1).unsqueeze(0)).cuda())
    fwd.remove()
    last_conv_output = np.squeeze(features[0].detach().cpu().numpy())
    
    masks = generate_masks_conv_output(input.transpose(0,1).shape[-3:], features[0], s=8, τ=τ)
    masks = masks.permute(3,0,1,2)
    N = len(masks)
    sal, weights, new_interactions, diff_interactions, pred_org = explain_SIDU(model, input, N, 0.5, masks, (16,224,224))
    if target is not None:
        return torch.tensor(sal)[target]
    
    return torch.tensor(sal)


def gradcamPAN(model, layer, input, target=None):
    features = []
    grad = []

    def fwd_hook(module, in_features, out_features, fout=None):
        fout += in_features

    def bck_hook(module, in_grad, out_grad, fout=None):
        fout += in_grad

    fwd = layer.register_forward_hook(partial(fwd_hook, fout=features))
    bck = layer.register_full_backward_hook(partial(bck_hook, fout=grad))

    output = model(norm(input).permute(1,0,2,3).cuda().unsqueeze(0))
    if target is not None:
        output[0, target].backward()
    else:
        output.max().backward()
    fwd.remove()
    bck.remove()
    features = features[0]
    grad = grad[0]

    α = grad.sum((2,3))
    cam = (torch.nn.functional.interpolate(
        torch.relu(features*α.view([*(α.shape), 1, 1])),
        [*(input.shape[-2:])],
        mode='bilinear',
        align_corners=True)
    ).sum(1)
    
    return cam.detach().cpu()


def risePAN(model, input, target=None, gpu_batch=24, N=1000, s=8, p1=0.1, data_length=4):
    B, CT, H, W = input.size()
    C = 3
    T = CT//(data_length * C)
    
    input_size = (T, H, W)
    cell_size = torch.ceil(torch.tensor(input_size) / s).long()
    up_size = ((s + 1) * cell_size).tolist()

    grid = (torch.rand(N, s//4, s, s) < p1).float()

    masks = torch.zeros((N, *input_size))
    shifts = []
    for i in range(N):
        t = torch.randint(0, cell_size[0], [1])
        x = torch.randint(0, cell_size[1], [1])
        y = torch.randint(0, cell_size[2], [1])
        shifts.append((t, x, y))
        masks[i, :, :] = F.interpolate(grid[[i]].unsqueeze(0).cuda(), up_size, mode='trilinear', align_corners=True).squeeze()[t:t + input_size[0], x:x + input_size[1], y:y + input_size[2]].cpu()
    masks = masks.reshape(-1, 1, *input_size)
    with torch.no_grad():
        p = []
        x = input.cuda()
        for i in range(0, N, gpu_batch):
            
            m = masks[i:i+gpu_batch].repeat_interleave(data_length,1).flatten(1,2).unsqueeze(2)
            batch = (m.cuda() * x.view(-1,C,H,W))
            batch = norm(batch.flatten(1,2))
            p.append(model(batch).cpu())
            del batch

        p = torch.cat(p)
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), masks.view(N, T * H * W))
        sal = sal.view((CL, T, H, W))
        sal = sal / N / p1
        if target is not None:
            sal = sal[target]

        return sal

#see https://github.com/satyamahesh84/SIDU_XAI_CODE.git
@torch.no_grad()
def siduPAN(model, layer, input, target=None, batch_size=64, data_length=4):
    model.eval()
    import numpy as np
    from scipy.spatial.distance import cdist
    from skimage.transform import resize
    
    def generate_masks_conv_output(input_size, last_conv_output, s=8):
        up_size = (torch.tensor(input_size) / s).ceil() * s
        masks = F.interpolate((last_conv_output > 0.5).float(), size=input_size[-2:], mode='bilinear', align_corners=True).permute(1,0,2,3)
        return masks
    
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))  

    def sim_differences(pred_org, preds):
        diff = abs(pred_org-preds)
        weights= kernel(diff,0.25)
        return weights, diff

    def uniqness_measure(masks_predictions):
        sum_all_cdist =(torch.cdist(masks_predictions, masks_predictions)).sum(axis=1)
        sum_all_cdist = norm(sum_all_cdist)
        return sum_all_cdist
    
    def explain_SIDU(model, inp, N, p1, masks, input_size):
        preds = []
        pred_org = model(norm(inp).cuda()).detach().cpu().numpy()

        with torch.no_grad():
            masks = masks.unsqueeze(2).float()
            inp_shape = inp.shape
            inp = inp.view(1, -1, 3, *(input_size[-2:])).cuda()
            for i in range(0, N, batch_size):
                m = masks[i:min(i+batch_size, N)].repeat_interleave(data_length,1)
                batch = norm(m.cuda() * inp).view([-1, *(inp_shape[1:])])
                preds.append(model(batch).detach().cpu())

            preds = torch.cat(preds).detach().cpu().numpy()
            weights, diff = sim_differences(pred_org, preds)
            interactions = uniqness_measure(torch.tensor(preds))
            new_interactions = interactions.reshape(-1, 1)
            diff_interactions = np.multiply(weights, new_interactions)
            sal = diff_interactions.numpy().T.dot(masks.squeeze().cpu().numpy().reshape(N, -1)).reshape(-1, *input_size)
            sal = sal / N / p1
        
        return sal
    
    
    features = []

    def fwd_hook(module, in_features, out_features, fout=None):
        fout += in_features

    fwd = layer.register_forward_hook(partial(fwd_hook, fout=features))
    output = model(norm(input).cuda())
    fwd.remove()
    last_conv_output = np.squeeze(features[0].detach().cpu().numpy())
    masks = generate_masks_conv_output([input.size(-3)//(3*data_length), *(input.shape[-2:])], features[0], s= 8)
    N = len(masks)

    sal = explain_SIDU(model, input, N, 0.5, masks, (8,224,224))
    if target is not None:
        return torch.tensor(sal)[target]
    
    return torch.tensor(sal)
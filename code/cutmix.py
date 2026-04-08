import torch
import numpy as np


class CutMix3D:
    def __init__(self, prop=1.0, beta=4.0):
        self.prop = prop
        self.beta = beta

    def __call__(self, tensor_list):
        # tensor_list: list of tensors to apply cutmix on
        if np.random.random() > self.prop:
            return tensor_list

        B, _, D, W, H = tensor_list[0].shape
        device = tensor_list[0].device
        indices = torch.randperm(B).to(device)
        
        lam = np.random.beta(self.beta, self.beta)
        cut_rat = np.power(1. - lam, 1/3)
        cut_d, cut_w, cut_h = int(D * cut_rat), int(W * cut_rat), int(H * cut_rat)
        cd, cw, ch = np.random.randint(D), np.random.randint(W), np.random.randint(H)

        d1, w1, h1 = np.clip(cd - cut_d // 2, 0, D), np.clip(cw - cut_w // 2, 0, W), np.clip(ch - cut_h // 2, 0, H)
        d2, w2, h2 = np.clip(cd + cut_d // 2, 0, D), np.clip(cw + cut_w // 2, 0, W), np.clip(ch + cut_h // 2, 0, H)

        output = []
        for t in tensor_list:
            mixed_t = t.clone()
            mixed_t[:, :, d1:d2, w1:w2, h1:h2] = t[indices, :, d1:d2, w1:w2, h1:h2]
            output.append(mixed_t)
        return output
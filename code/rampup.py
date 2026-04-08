import numpy as np
from config import Config

''' 

    Original Source : https://arxiv.org/abs/1610.02242
    
'''

def sigmoid_rampup(current, rampup_length):

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, config: Config):

    consistency = config.consistency
    consistency_rampup = config.consistency_rampup
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

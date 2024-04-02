
"""
This file contains some functions and classes which can be useful in very diverse projects.
"""

import os
import sys
import torch
import random
import logging
import traceback
import numpy as np
from os.path import join


def make_deterministic(seed=0):
    """
    Make results deterministic. If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

def load_settings(config_file: str):
    # Load configuration
    cfg = OmegaConf.load(config_file)

    # resolve interpolation
    OmegaConf.resolve(cfg)

    return cfg
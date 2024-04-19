# -*- coding: utf-8 -*-
import os
import shutil
import pathlib
import glob

import torch
import dgl
from dgl.data.utils import save_graphs
import json
from scipy.sparse import csr_matrix
import numpy as np

import sys
sys.path.append('..')
from .macro import *

def output_z(z_main, z_sub):
    json_root = {}
    json_root["main_feature"] = z_main.tolist()
    json_root["sub_feature"] = z_sub.tolist()
    
    output_path = pathlib.Path("/home/zhang/datasets_v3_1_quater/val")
    file_name = "latent_z.json"
    binfile_path = os.path.join(output_path,file_name)

    with open(binfile_path,'w',encoding='utf-8')as fp:
        json.dump(json_root,fp, indent=4)
        
        

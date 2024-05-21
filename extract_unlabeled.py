import os
import random
import shutil
from shutil import move
import numpy as np

seed = 2022


def unlabeled_split(src_folder, tgt_folder, unlabeled_size=437):
   src_input_dir = os.path.join(src_folder, 'input')
   src_gt_dir = os.path.join(src_folder, 'GT')
   
   tgt_input_dir = os.path.join(tgt_folder, 'input')
   
   if not os.path.exists(tgt_input_dir):
       os.mkdir(tgt_input_dir)

   files = [f for f in os.listdir(src_input_dir)]
   
   np.random.seed(seed)
   indices = np.random.permutation(len(files))
   
   selected = indices[:unlabeled_size]
   
   for s in selected:
       move(os.path.join(src_input_dir, files[s]), os.path.join(tgt_input_dir, files[s]))
       os.remove(os.path.join(src_gt_dir, files[s]))
   
dataset = "suid"
src_folder = f"/home/ddimauro/Neptune/Enhancement/Semi-UIR/data/{dataset}/labeled"
target_folder = f"/home/ddimauro/Neptune/Enhancement/Semi-UIR/data/{dataset}/unlabeled"
    
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

unlabeled_split(src_folder, target_folder)

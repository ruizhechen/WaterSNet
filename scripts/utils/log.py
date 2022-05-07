import logging
import os
import datetime
import yaml
import glob
import shutil
import torch
from tensorboardX import SummaryWriter
def get_exp_name():
    exp_name="exp_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(os.path.join("./runs", exp_name)):
        os.makedirs(os.path.join("./runs", exp_name))
        os.makedirs(os.path.join("./runs", exp_name,"checkpoints"))
        os.makedirs(os.path.join("./runs", exp_name,"eval_results"))
        os.makedirs(os.path.join("./runs", exp_name,"logs"))
        os.makedirs(os.path.join("./runs", exp_name,"runs"))
        os.makedirs(os.path.join("./runs", exp_name,"scripts"))
    return exp_name
def summary_writer(exp_name):
    writer=SummaryWriter(os.path.join("../runs"))
    return writer
def get_logger(exp_name, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler("../logs/log", "w") 
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
def save_ckpt(ckpt,exp_name,**kwargs):
    torch.save(ckpt,os.path.join("../checkpoints", f"{kwargs['net_name']}_epoch_{kwargs['epoch']}_itr_{kwargs['ite_num']}.pth"))


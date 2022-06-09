import os
import sys
import copy
import numpy as np
import json
from tqdm import tqdm
from torchvision import transforms
import torch
from torch import nn
import fire
sys.path.append(os.path.join("Preprocesamiento"))
from Preprocesamiento import Custom_Transforms
from Preprocesamiento.dataLoader import belugaDataset

sys.path.append("Models")

from Train_utils.Multi_parameter_train import multi_parameter_training
from Generation_utils.utils import parallel_gen_metadata_model

def main():
    res_dir=os.path.join("..","Results","VAE","Basic_CNN_EDM","Test_1")
    model_state=torch.load(os.path.join(res_dir,'best0.pt'))

    if "metadata" not in os.listdir(res_dir):
        os.mkdir(os.path.join(res_dir,"metadata"))
    meta_dir=os.path.join(res_dir,"metadata")

    mpt=multi_parameter_training(
                results_directory=res_dir,
                dataset_root_directory=os.path.join("..","belugas_classification"),
                train=True,
                test=True,
                K_fold_training=None,
                visualization=False
            )
    
    
    test_json=json.load(open(os.path.join(res_dir,"config.json")))
    test_json_save=copy.deepcopy(test_json)
    trainer_args=test_json["trainer"]
    model_args=test_json["model"]
    transforms_args=test_json["transforms"]

    mpt.prepare_transforms(transforms_args)
    mpt.set_datasets(None,False)
    mpt.set_model(model_args)
            
    if test_json["trainer"]["use_cuda"]:
        trainer_args["model"]=mpt.instantiated_model.cuda()
    else:
        trainer_args["model"]=mpt.instantiated_model

    mpt.instantiated_model.load_state_dict(model_state)

    print(len(mpt.dataset))

    parallel_gen_metadata_model(data_base=mpt.dataset,
                                out_meta_dir=meta_dir,
                                model=mpt.instantiated_model,
                                batch_size=20,
                                num_workers=6,
                                args=['x'],
                                device_in='cuda')

    
if __name__ == "__main__":
    fire.Fire(main)
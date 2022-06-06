import os
import sys
import copy
import numpy as np
import json
from tqdm import tqdm
from torchvision import transforms
import torch
from torch import nn

sys.path.append(os.path.join("..","Preprocesamiento"))
from Preprocesamiento import Custom_Transforms
from Preprocesamiento.dataLoader import belugaDataset

sys.path.append(os.path.join("..","Models"))
from Models import DL_utils
from Models import Encoding_Decoding_modules
import Models

from Train_utils.Multi_parameter_train import multi_parameter_training
from Generation_utils.utils import parallel_gen_metadata_model

def main():
    res_dir=os.path.join("..",'GMVAE_A1_3')
    model_state=torch.load(os.path.join(res_dir,'best0.pt'))

    DB="/home/liiarpi-01/proyectopaltas/Local_data_base/Data_Base_v2"
    meta_dir="/home/liiarpi-01/proyectopaltas/Local_data_base/metadata_GMVAE_A1_3"
    model=os.listdir(os.path.join("..","Results",model))
    EncDec=os.listdir(os.path.join("..","Results",EncDec))
    test="" #Test dir direction

    mpt=multi_parameter_training(
                results_directory=os.path.join("..","Results",model,EncDec),
                dataset_root_directory=os.path.join("..","belugas_classification"),
                train=True,
                test=True,
                K_fold_training=None,
                visualization=False
            )
    
    
    test_json=json.load(open(os.path.join(test,"config.json")))
    test_json_save=copy.deepcopy(test_json)
    trainer_args=test_json["trainer"]
    model_args=test_json["model"]
    transforms_args=test_json["transforms"]

    mpt.prepare_transforms(transforms_args)
    mpt.set_datasets(test,False)
    mpt.set_model(model_args)
            
    if test_json["trainer"]["use_cuda"]:
        trainer_args["model"]=mpt.instantiated_model.cuda()
    else:
        trainer_args["model"]=mpt.instantiated_model
    
    #mpt.Train()

    #model=GMVAE(image_dim=int(512),
    #    image_channels=1,
    #    repr_sizes=[3,6,12,24,48],
    #    layer_sizes=[200,100,50],
    #    w_latent_space_size=10,
    #    z_latent_space_size=10,
    #    y_latent_space_size=5,
    #    conv_kernel_size=7,
    #    conv_pooling=False,
    #    activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
    #    conv_batch_norm=True,
    #    NN_batch_norm=True,
    #    stride=2,
    #    device="cpu")
    #model.to("cuda")

    mpt.instantiated_model.load_state_dict(model_state)

    #d_tt=transforms.Compose([
    #    ndvi_desc(),
    #    multi_image_resize(ImType=['SenteraNDVI'],size=(512,512)),
    #    multi_ToTensor(ImType=['SenteraNDVI']),
    #    select_out_transform(selected=['SenteraNDVI','Place','Date','landmarks'])### ----------------------------------------
    #    ])
#
    #datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)
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
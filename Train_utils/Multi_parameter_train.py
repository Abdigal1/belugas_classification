import os
import sys
import numpy as np
import json
from tqdm import tqdm
from torchvision import transforms
import torch
from torch import nn

sys.path.append(os.path.join("..","Data_preprocessing"))
import Custom_Transforms
from Custom_dataloader import segmentationDataset

sys.path.append(os.path.join("..","Models"))
#from Encoding_Decoding_modules.Basic_Encoding_Decoding_Module import Basic_Convolutional_EDM
import Encoding_Decoding_modules

#from Models.CVAE_NETs.P_NET import P_NET
#from Models.CVAE_NETs.Q_NET import Q_NET

#from Models.pyro_CVAE import CVAE
import Models

from TT_pyro_class import trainer

class multi_parameter_training(object):
    def __init__(self,results_directory,dataset_root_directory,train=True,test=True,K_fold_training=None,visualization=False):

        self.datasets={}
        self.Compose_trans=None
        self.results_directory=results_directory
        self.dataset_root_directory=dataset_root_directory
        self.train=train
        self.test=test
        self.K_fold_training=K_fold_training

        self.test_dirs=np.array(os.listdir(self.results_directory))
        self.test_dirs=np.vectorize(lambda td:os.path.join(self.results_directory,td))(self.test_dirs)
        #self.test_dirs=np.vectorize(lambda td:os.path.join(td,"config.json"))(self.test_dirs)
    
    
    def prepare_transforms(self,transform_args):
        trans_seq=list(transform_args.keys())
        instantiated_trans_seq=[]
        for t in trans_seq:
            instantiated_trans_seq.append(getattr(Custom_Transforms,t)(**(transform_args[t])))
        self.Compose_trans=transforms.Compose(instantiated_trans_seq)
 
    def set_datasets(self):
        segmentation_dataset_dir=self.dataset_root_directory
        annotations_dir=os.path.join(segmentation_dataset_dir,"annotations")
        #TODO: load validation and other splits
        train_ann=json.load(open(os.path.join(annotations_dir,"train.json")))
        test_ann=json.load(open(os.path.join(annotations_dir,"test.json")))
        self.datasets["train_set"]=segmentationDataset(train_ann,segmentation_dataset_dir,transform=self.Compose_trans)
        self.datasets["test_set"]=segmentationDataset(test_ann,segmentation_dataset_dir,transform=self.Compose_trans)

            
    def parse_activators(self,raw_params):
        for param in list(raw_params.keys()):
            if "activators" in param:
                if isinstance(raw_params[param]["name"],list):
                    instantiated_act=[]
                    for act_id in range(len(raw_params[param]["name"])):
                        instantiated_act.append(
                            getattr(nn,raw_params[param]["name"][act_id])(**(raw_params[param]["params"][act_id]))
                        )
                else:
                    instantiated_act=getattr(nn,raw_params[param]["name"])(**(raw_params[param]["params"]))
                raw_params[param]=instantiated_act
        return raw_params

    def set_model(self,model_args):
        submodels_data=model_args["sub_modules"]
        #Load modules
        for module in list(submodels_data.keys()):
            if module=="P_NET" or module=="Q_NET":
                #Load variational modules
                inst_module=getattr(
                    getattr(
                        getattr(
                            Models,
                            submodels_data[module]["variational_generation_type"]
                            ),
                        module
                    ),
                    module
                    )(**(self.parse_activators(submodels_data[module]["parameters"])))
                submodels_data[module]=inst_module
            else:
                #Load variational modules
                inst_module=getattr(
                    getattr(
                        Encoding_Decoding_modules,
                        "Basic_Encoding_Decoding_Module"
                    ),
                    submodels_data[module]["module_type"]
                    )(
                        **(self.parse_activators(submodels_data[module]["parameters"]))
                        )
                submodels_data[module]=inst_module
            
            #Instance model
        model_args["model_params"].update(model_args["sub_modules"])
        self.instantiated_model=getattr(
            getattr(
                Models,
                model_args["model_class"]
                ),
                model_args["model_name"]
                )(**(model_args["model_params"]))

        
    def Train(self):
        for test_id in tqdm(range(len(self.test_dirs)),desc="Model test"):
            test=self.test_dirs[test_id]
            test_json=json.load(open(os.path.join(test,"config.json")))
            trainer_args=test_json["trainer"]
            model_args=test_json["model"]
            transforms_args=test_json["transforms"]

            self.prepare_transforms(transforms_args)
            self.set_datasets()
            self.set_model(model_args)
            
            if test_json["trainer"]["use_cuda"]:
                trainer_args["model"]=self.instantiated_model.cuda()
            else:
                trainer_args["model"]=self.instantiated_model
            trainer_args["data_dir"]=test
            trainer_args["dataset"]=self.datasets

            self.trainer=trainer(**(trainer_args))
            
            if test_json["experiment_state"]=="waiting":
              try:
                if self.train and self.test:
                  self.trainer.optimizer=torch.optim.Adam(self.trainer.model.parameters(),**(test_json["optimizer"]))
                  self.trainer.train_test(**(self.datasets))
                  test_json["experiment_state"]="done"
                elif self.train and not(self.test):
                  #set train rutine
                  self.trainer.optimizer=torch.optim.Adam(self.trainer.model.parameters(),**(test_json["optimizer"]))
                  test_json["experiment_state"]="done"
                elif not(self.train) and self.test:
                  #set optimizer
                  self.trainer.train(self.datasets["train"])
                  self.trainer.test(self.datasets["test"])
                  test_json["experiment_state"]="done"
                #elif self.K_fold_training!=None:

              except Exception as e:
                tqdm.write("tranning failed")
                tqdm.write(e)
                test_json["experiment_state"]="fail"
                test_json["error"]=e
                #TODO show error
              #save config.json
              f=open(os.path.join(test,"config.json"),"wb")
              f.write(json.dump(test_json))
              tqdm.write("Training model "+str(test_id))
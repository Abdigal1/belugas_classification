import fire
import os
import sys
#sys.path.append('/content/MMSports_Challenge')
sys.path.append("Models")
#sys.path.append('/content/MMSports_Challenge/Data_preprocessing/')
#sys.path.append('/content/MMSports_Challenge/Train_utils/')

from Train_utils.Multi_parameter_train import multi_parameter_training




def main():


    #mpt=multi_parameter_training(
    #results_directory=os.path.join("..","Results","VAE","Basic_CNN_EDM"),
    #dataset_root_directory=os.path.join("..","belugas_classification"),
    #train=True,
    #test=True,
    #K_fold_training=None,
    #visualization=False
    #)
#
    #mpt.Train()

    for model in os.listdir(os.path.join("..","Results")):
        for EncDec in os.listdir(os.path.join("..","Results",model)):
            mpt=multi_parameter_training(
                results_directory=os.path.join("..","Results",model,EncDec),
                dataset_root_directory=os.path.join("..","belugas_classification"),
                train=True,
                test=True,
                K_fold_training=None,
                visualization=False
            )

            mpt.Train()


if __name__=="__main__":
    fire.Fire(main)
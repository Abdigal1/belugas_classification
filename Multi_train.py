import fire

#sys.path.append('/content/MMSports_Challenge')
#sys.path.append('/content/MMSports_Challenge/Models/')
#sys.path.append('/content/MMSports_Challenge/Data_preprocessing/')
#sys.path.append('/content/MMSports_Challenge/Train_utils/')

from Train_utils.Multi_parameter_train import multi_parameter_training




def main():


    mpt=multi_parameter_training(
    results_directory="/../../Results/Basic_CNN_EDM",
    dataset_root_directory="/../DataBase/instance-segmentation-challenge/",
    train=True,
    test=True,
    K_fold_training=None,
    visualization=True
    )

    mpt.Train()


if __name__=="__main__":
    fire.Fire(main)
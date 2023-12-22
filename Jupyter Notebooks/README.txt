This file contains five sections:

SEC0: Contains the details of materials provided in the zip file.
SEC1: Contains directions to run the models in case the reviewers use the provided optuna trained models.
SEC2: Contains directions to train the optuna models and save the pickle file. Afterwards, please follow SEC1 for hyper local tuning.
SEC3: Contains directions for gurobi license. PLEASE READ CAREFULLY.  
SEC4: Provide details of the major libraries used, along with their versions.

=============================== SEC0 => Folder Details ===============================

Colab_Notebooks        -> Contains four colab notebooks corresponding to optuna training and hyper local tuning. Details are given in sections below.

ResNet50_Optuna_models -> Contains four pickle files for optuna trained models with ResNet50 setting. The path of these files is required in "OPTUNA_MODEL_DIRECTORY" 
				  variable in ResNet50_HLS.ipynb notebook. 

				Note: this folder can be accessed from drive folder "https://drive.google.com/drive/folders/16nLo4vAecD6kphYx3Pa5DYE0vciyiZpD?usp=sharing" . Please open in incognito mode in your browser and download the data.  

CIFAR_10_20K           -> Contains the data files of CIFAR-10 dataset. Training, validation and testing data files provided are the ones used in our experiments.


Note: The Appendix is also provided along with above four folders.

============================= SEC1 => Directions: With optuna Models =============================


There are two models for which notebooks are provided.

For Simple CNN Models     : CNN_HLS.ipynb
For ResNet50 based Models : ResNet50_HLS.ipynb

NOTE: In given notebooks, the specifications are already set to the corresponding models. In first cell, the libraries will be imported.

The following directions are for both type of models:

1] In the second cell, please provide the directories of the data files, optuna trained model file, and full output directory with excel(.xlsx) file name.

2] In addition, please specify a gurobi environment to run the solver. The model is too large for free version, so atleast academic license 
   is required. In case you do not have access to licese or have any other issues, please see section 3("GUROBI LICENSE") provided below.

3] Finally, run all the remaining cells, the output is printed as log of final cell and as excel file in the output directory.

4] In case of CNN_HLS.ipynb, please mention the number of dense layers of the model. (3 or 5)



NOTE1: The first row in output file is the result of optuna's optimal model t = 0.

NOTE2: If you are running this code on workstation other than colab pro plus, then please note that the runtimes will not match
	to the studies we have provided, as the provided optuna models are trained on colab pro plus.


=========================== SEC2 => Directions: Without optuna Models ==========================================



For both models, notebooks are provided to generate an optima optuna mode.

For Simple CNN Models     : CNN_OPTUNA.ipynb
For ResNet50 based Models : ResNet50_OPTUNA.ipynb


NOTE: In given notebooks, the specifications are already set to the corresponding models. In first cell, the libraries will be imported.


The following directions are for both type of models:

1] In the second cell, please provide the directories of the data files and output directory to save the pickle file in "OPTUNA_MODEL_DIRECTORY" variable.

2] In addition, please specify the type of sampler ( 'grid', 'random', 'qmc', 'tpe' ) and number of optuna trials (already set to match our study).

3] Finally, run all the remaining cells, the output is printed as pickle file in OPTUNA_MODEL_DIRECTORY.

4] In case of CNN_HLS.ipynb, please mention the number of dense layers of the model. (3 or 5)

5] After you have the pickle file, please follow steps of SEC1 for hyper local tuning.


=================================== SEC3 => GUROBI LICENSE (I HAVE DELETED MY CREDENTIALS) ==================================


# Warning: I am providing my license in case reviewers have problem getting the license. Please run the code with OutputFlag = 0, else my email id with academic license will be revealed in the log, 
# this might be a violation to the double blind policy. We request reviewers to use their own academic license as it is easily available on institute ids.

ENV = gp.Env( empty=True )
ENV.setParam( 'WLSACCESSID', ' ' )   
ENV.setParam( 'WLSSECRET', ' ' )    
ENV.setParam( 'LICENSEID', )
ENV.setParam( 'OutputFlag', 0 )      # To Turn-off Logs
ENV.start()


====================================== SEC4 => LIBRARIES ======================================

Below are the main libraries and their versions on which the code has been run successfully.

Pandas     -> 1.5.3
Numpy      -> 1.22.4
Tensorflow -> 2.12.0
Gurobipy   -> 10.0.1
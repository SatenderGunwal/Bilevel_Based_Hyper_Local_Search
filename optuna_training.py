# This file contains the code for the initial training of the model using OPTUNA. Two type of models were used, a CNN and Pretrained ResNet50 for CIFAR-10 dataset. 
# The code will need tweaking of certain parameters depending on the model in use. Please change the size of the model(dense layers) depending on the space and 
# computing power availability. Caution: Code is problem specific and written in a simple manner.

# !pip install optuna
import tensorflow as tf
import optuna

import numpy as np
import pickle as pkl
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Importing functions needed from "helper_functions.py"
from helper_functions import data_loader
from helper_functions import CNN
from helper_functions import CNN_ResNet50 # Use it when model is resnet50.
    





@tf.function
def loss_function_optuna( y_dataset, logits, loss ): # logits = model(x_dataset)
    total_loss = loss(y_dataset, logits)
    total_loss = tf.cast( total_loss, dtype=tf.float32 )
    return total_loss

def fmin_loss( model, loss_function, optimizer, batch_size , epochs, record = True ):  # lamda, not exp(lamda), Works with both tf.Variable and tf.constant type lambda input, (or just scalar)
    
    train_df = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_df = train_df.shuffle(buffer_size = 1024).batch(batch_size)

    All_Epoch_Gradients, All_Epoch_Weights = [], []

    for epoch in range(epochs):

        weights0 = [var.numpy() for var in model.trainable_weights] # Getting only trainable weights at which the gradient is being calculated.
        # Note : model.get_weights() retrieves all the weights (including non-trainable)

        Step_Gradient, Num_batch = [], 0
        
        for step,(x_train_,y_train_) in enumerate(train_df):
       
            with tf.GradientTape(persistent = True) as tape:

                logits = model(x_train_, training=True)
                total_loss1 = loss_function_optuna( y_train_, logits, loss = loss_function ) 

            vars_list = model.trainable_weights
            grads = tape.gradient(total_loss1, vars_list)      # for ref  - https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough 
            optimizer.apply_gradients(zip(grads,vars_list))

            if record : 
                if step == 0 : 
                    Step_Gradient = grads
                else:
                    for idx in range(len(Step_Gradient)):
                        Step_Gradient[idx] =  tf.add(Step_Gradient[idx], grads[idx])
            Num_batch = step
        
        if record : 
            Step_Gradient = [ i/Num_batch for i in Step_Gradient ] 
            All_Epoch_Gradients.append( Step_Gradient )
            All_Epoch_Weights.append(weights0)      
    if record :
        return All_Epoch_Gradients, All_Epoch_Weights
    else: 
        return 0, 0
    
def optuna_optimizer(trial):

    tf.keras.backend.clear_session()

    alphas = [ trial.suggest_float(f'regularization{i}', 1e-6, 1e-1, log=True) for i in range(NUMBER_OF_DENSE_LAYERS) ]

    # Define the new model

    layer_info_ = [ {'type': 'dense', 'params': {'units': 64, 'activation': 'relu', 'kernel_regularizer':tf.keras.regularizers.l2(i)}} for i in alphas[:-1]]
    layer_info_ += [ {'type': 'dense', 'params': {'units': output_shape, 'activation': 'softmax', 'kernel_regularizer':tf.keras.regularizers.l2(alphas[-1])}} ]

    model = CNN( input_shape, output_shape ).generate_model( layer_info_ ) # NOTE_: Use "CNN_ResNet50" class from helper_functions.py for resnet model
    
    # Optimizing
    optimizer         = tf.keras.optimizers.Adam()
    loss_function     = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    All_Epoch_Gradients, All_Epoch_Weights = fmin_loss( model, loss_function, optimizer,  128,  100, True) # We took 10 epochs instead of 100 for resnet50

    # Getting Scores
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    sca = tf.keras.metrics.SparseCategoricalAccuracy()

    y_pred_train = model.predict(x_train)
    train_loss_unregularized = cce(y_train, y_pred_train).numpy()
    train_acc = sca(y_train, y_pred_train).numpy()

    y_pred_val = model.predict(x_val)
    val_loss_unregularized = cce(y_val, y_pred_val).numpy()
    val_acc = sca(y_val, y_pred_val).numpy()

    y_pred_test = model.predict(x_test)
    test_loss_unregularized = cce(y_test, y_pred_test).numpy()
    test_acc = sca(y_test, y_pred_test).numpy()

    print("\nTraining:  Loss ", train_loss_unregularized, "  Accuracy", train_acc*100 )
    print("Validation: Loss ", val_loss_unregularized, "  Accuracy", val_acc*100 )
    print("Test:       Loss", test_loss_unregularized, "  Accuracy", test_acc*100, "\n\n")

    with open("{}.pickle".format(trial.number), "wb") as fout:
        pkl.dump(model, fout)

    with open("training_info_{}.pickle".format(trial.number), "wb") as fout:
        Dict_ = { "Gradients" : All_Epoch_Gradients, "Weights" : All_Epoch_Weights }
        pkl.dump(Dict_, fout)
    
    score = val_loss_unregularized
    return(score)


def optuna_training( num_trials ):

    time1 = datetime.now()

    if TYPE_OF_SAMPLER == 'grid': 
        search_space = { f"regularization{i}" : list(np.linspace(1e-6,1e-1,10)) for i in range(NUMBER_OF_DENSE_LAYERS) }
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction = "minimize")
    elif TYPE_OF_SAMPLER == 'random': 
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), direction = "minimize")
    elif TYPE_OF_SAMPLER == 'qmc': 
        study = optuna.create_study(sampler=optuna.samplers.QMCSampler(), direction = "minimize")
    elif TYPE_OF_SAMPLER == 'tpe': 
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction = "minimize")
    else:
        print("...Mention the correct sampler name...")

    study.optimize(optuna_optimizer, n_trials = num_trials)

    print('\n\n')
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))

    print( "\n\n ", "Trial Number: ", trial.number, "\n" )

    time2 = datetime.now()
    delta = time2 - time1
    print(f"Time difference is {delta.total_seconds()} seconds")

    # Loading Best OPTUNA model to get initial feasible weights for trainable layers
    with open("{}.pickle".format(trial.number), "rb") as fin:
        best_clf = pkl.load(fin)

    with open("training_info_{}.pickle".format(trial.number), "rb") as fin_:
        wt_grad = pkl.load(fin_)

    # Getting Optimal Model's Weights and Gradients values for Hessian Approximations
    weight_sets = wt_grad['Weights']
    grad_sets   = wt_grad['Gradients']

    # Getting Optimal Model's Weights and Hyperparameters
    full_weights_list          = best_clf.get_weights()
    trainable_weights_list     = best_clf.trainable_weights
    init_hyperparameters = [ tf.Variable(value) for key, value in trial.params.items() ]

    return trainable_weights_list, full_weights_list, init_hyperparameters, weight_sets, grad_sets, delta
    

if __name__=="__main__":

    #---------- GOLBAL VARIABLES NEEDED TO RUN THE EXPERIMENT --------------

    # 1] ENTER DATA DIRECTORIES
    base_dir  = "/content/drive/MyDrive/CIFAR_Dataset/CIFAR_10/Batch3_1+10K"
    train_dir = base_dir+'/cifar10_train.npz'
    val_dir   = base_dir+'/cifar10_val.npz'
    test_dir  = base_dir+'/cifar10_test.npz'
    # 2] ENTER DIRECTORY TO SAVE OPTUNA's TRAINED MODEL
    OPTUNA_MODEL_DIRECTORY = "/content/drive/MyDrive/CIFAR_Dataset/CIFAR_10/cnn_optuna.pickle"

    NUMBER_OF_DENSE_LAYERS = 3 # or 5; Includes the output layer.
    TYPE_OF_SAMPLER = 'grid' #  'grid', 'random', 'qmc', 'tpe' 
    NUM_OPTUNA_TRIALS = 10

    ## ============================================  (FOR CNN MODEL)LOADING DATA  ==========================================================
    x_train, x_val, x_test, y_train, y_val, y_test = data_loader( train_dir, val_dir, test_dir, model_type='cnn' )
    input_shape = 32
    output_shape = 10
    ## ============================================  (FOR ResNet50 MODEL)LOADING DATA  =====================================================
    # x_train, x_val, x_test, y_train, y_val, y_test = data_loader( train_dir, val_dir, test_dir, model_type='resnet50' )
    # input_shape  = x_train.shape[1:]
    # output_shape = 10
    # NOTE_ :: In optuna_optimizer() function, use CNN_ResNet50 class from helper_function.py instead of CNN for resnet model.

    # ====================== OPTUNA TRAINING ====================
    trainable_weights_list, full_weights_list, init_hyperparameters, weight_sets, grad_sets, optuna_time = optuna_training(NUM_OPTUNA_TRIALS)

    model__ = [ trainable_weights_list, full_weights_list, init_hyperparameters, weight_sets, grad_sets, optuna_time ]
    # LOADING OPTUNA TRAINED MODELS
    with open(OPTUNA_MODEL_DIRECTORY, "wb") as fout:
            pkl.dump(model__, fout)

    # SAVE THIS TRAINED MODEL FILE AND USE FOR THE HYPER LOCAL TUNING PART. ALREADY TRAINED FILES GENERATED FROM THIS MODEL ON 
    # FIXED DATASETS ARE ALSO PROVIDED IN THE FOLDER WITH JUPYTER NOTEBOOK FILES.


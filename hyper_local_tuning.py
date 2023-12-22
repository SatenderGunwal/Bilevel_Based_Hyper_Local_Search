
# !pip install gurobipy
# !pip install tensorflow
import gurobipy as gp
from gurobipy import GRB

import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime
import math

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

# Importing functions needed from "helper_functions.py"
from helper_functions import data_loader
from helper_functions import CNN
from helper_functions import CNN_ResNet50 # Use it when model is resnet50.
from helper_functions import layer_information
from helper_functions import remaining_approximation

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@tf.function
# Layer_Weights_ToRegularize : Only give list of weights to be used on regularization term. (Do not include bias weights)
def loss_function( y_dataset, logits, Layer_Weights_ToRegularize = None,
                  Regularization_Parameters = None, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ): # logits = model(x_dataset)

    total_loss = loss(y_dataset, logits)
    total_loss = tf.cast( total_loss, dtype=tf.float32 )

    if Regularization_Parameters == None or Layer_Weights_ToRegularize == None:  return total_loss
    for weight, lamda in zip( Layer_Weights_ToRegularize, Regularization_Parameters ): 

        if not tf.is_tensor(weight): tf.convert_to_tensor(weight)

        regularization  = lamda * tf.reduce_sum(tf.square(weight))   # Element-wise square -> Adding all terms -> Multiply by lamda
        regularization /= tf.cast(tf.size(weight), dtype=tf.float32) # Get number of parameters (N)
        total_loss     += regularization
        
    return total_loss


# COMPUTING VALIDATION LOSS COEFFICIENTS
def gradient_validation( batch_size = 32 ):

    val_df = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    val_df = val_df.shuffle(buffer_size = 1024).batch(batch_size)

    Step_Gradient, Num_batch = [], 0
    for step,(x_t,y_t) in enumerate(val_df):
        with tf.GradientTape(persistent=True) as tape:

            logits = model_(x_t)
            loss_  = loss_function( y_t, logits )

        vars_list = model_.trainable_weights
        grads = tape.gradient(loss_, vars_list)

        if step == 0 : Step_Gradient = grads
        else:
            for idx in range(len(Step_Gradient)):
                Step_Gradient[idx] =  tf.add(Step_Gradient[idx], grads[idx])
        Num_batch = step

    Step_Gradient = [ i/Num_batch for i in Step_Gradient ] 
    return Step_Gradient

# SYMMETRIC RANK-1 HESSIAN APPROXIMATION 
def sr1_hessian_approximation( weight_sets, grad_sets  ):

    trainable_count = sum(tf.keras.backend.count_params(weights) for weights in model_.trainable_weights)
    B_k             = np.identity(trainable_count, dtype = 'float32')

    weight_sets, grad_sets = weight_sets[:50], grad_sets[:50]

    total_iterations = len(weight_sets) - 1
    total_weights    = len(grad_sets[0])

    for iter in range( 1, total_iterations + 1  ):

        y_k, s_k = [], []
        for idx in range( total_weights ):

            y_k.append( (grad_sets[iter][idx] - grad_sets[iter-1][idx]).numpy() )
            s_k.append( weight_sets[iter][idx] - weight_sets[iter-1][idx] )

        arr1, arr2 = y_k[0].reshape(-1), s_k[0].reshape(-1)
        for tensor1, tensor2 in zip( y_k[1:], s_k[1:] ):
            arr1 = np.concatenate((arr1, tensor1), axis=None)
            arr2 = np.concatenate((arr2, tensor2), axis=None)
        del(y_k, s_k)
        Bk_sk = B_k.dot(arr2)
        term0 = arr1 - Bk_sk
        del(Bk_sk)
        TERM2 = np.outer( term0, term0 ) / term0.dot(arr2)
        del(term0)
        B_k  +=  TERM2
    return B_k

# =========================== DIRECTION PROBLEM =========================================

# Gurobi optimization
def Bilevel_Descent_Direction( GradUObj, Hessian_Follower, delta ): 

    (Rows_, Columns_)        = Hessian_Follower.shape
    Num_regularization_param = len(init_hyperparameters)
    
    # MODEL AND VARIABLE DECLARATION
    m = gp.Model(env = ENV )

    Model_Variables = m.addMVar( (Columns_), lb = -1, ub = 1, vtype = 'C' )
    # Note: Coefficients are scaled to avoid numerical issues.
    m.setObjective( 1e+6 * (GradUObj @ Model_Variables[:-Num_regularization_param]), 1 )
    m.addConstr( 1e+5 * (Hessian_Follower @ Model_Variables) <= delta*1e+5 )
    m.addConstr( 1e+5 * (Hessian_Follower @ Model_Variables) >= - delta*1e+5 )
    # OUTPUT HANDLING
    try:
        m.optimize()
        return m.X, m.ObjVal, m.Runtime
    except gp.GurobiError:
        m.computeIIS()
        m.write("IIS_System.ilp")
        return "Error in LB : GurobiError :: ", m.status

# Function to combine the hessian data and giving submatrix with random rows.
def random_constraints(percent_rows_used):

    hessian_part1     = sr1_hessian_approximation( weight_sets, grad_sets )
    remaining_columns = remaining_approximation( trainable_weights_list, init_hyperparameters )
    # hessian_full      = tf.concat( [ hessian_part1, tf.transpose(remaining_columns) ], axis = 1 )
    hessian_full      = np.concatenate((hessian_part1, remaining_columns.numpy().T),axis=1)
    del(hessian_part1, remaining_columns)
    num_hyperparams = len(init_hyperparameters)
    total_rows     = hessian_full.shape[0]

    # ======= PRE-PROCESSING ========
    imp_hessian_full = []
    for row in hessian_full:
        if list(row[-num_hyperparams:])!=[0 for i in range(num_hyperparams)]:
            imp_hessian_full.append(row)
    del(hessian_full)

    imp_hessian_full      = np.array(imp_hessian_full)
    Non_zero_rows         =  len(imp_hessian_full)
    percent_rows_remained =  Non_zero_rows/total_rows
    # print( f"\nAfter Pre-processing rows remained :: {percent_rows_remained*100} percent\n" )
    if percent_rows_remained <= percent_rows_used:
        return imp_hessian_full
    else:
        rows_asked     = percent_rows_used*total_rows
        np.random.shuffle(imp_hessian_full)     
        # Rows are shuffled. Each row remains unchanged.
        imp_hessian_full = imp_hessian_full[ : int(rows_asked), : ]

    return imp_hessian_full

def unflatten( full_weight_direction ): # Converts flattened directions into weight shapes
    result = []
    start = 0
    for param_size, shape in zip( layer_wise_params, layer_wise_shapes ):

        end = start + param_size[0]
        flat_list_params = np.array(full_weight_direction[start:end])
        start = end

        # Converting to tensor object just to use tf.reshape() function.
        flat_list_params = tf.convert_to_tensor(flat_list_params)
        flat_list_params = tf.reshape(flat_list_params, list(shape))
        result.append( flat_list_params )
    return result

def loss_value( new_weights, new_hyperparams, full_old_weights, Without_GSS = True ): # At every new point, gives loss value by using separate loss object

    new_hyperparams = [ float(p.numpy()) for p in new_hyperparams ]
    layer_info_ = layer_information( output_shape, hidden_dense_layers, dense_kernel_regularizers=new_hyperparams[:-1], output_kernel_regularizer=new_hyperparams[-1], include_flatten=False )
    
    model = CNN( input_shape, output_shape).generate_model( layer_info_)  # **USE CNN_ResNet50 FOR RESENET MODEL**

    # Set the new weights as the model's weights. Non-trainable weights fixed as in optuna training.
    for idx, wt in zip(index_set,new_weights):
        full_old_weights[idx] = wt
    model.set_weights(full_old_weights)

    # Getting Scores
    cce = tf.keras.losses.SparseCategoricalCrossentropy() 
    y_pred_val = model.predict(x_val)
    val_loss_unregularized = cce(y_val, y_pred_val).numpy()

    if Without_GSS:
        # Remaining LOSS
        y_pred_train = model.predict(x_train)
        train_loss_unregularized = cce(y_train, y_pred_train).numpy()
        y_pred_test = model.predict(x_test)
        test_loss_unregularized = cce(y_test, y_pred_test).numpy()
        # Accuracy
        sca = tf.keras.metrics.SparseCategoricalAccuracy()
        train_acc = sca(y_train, y_pred_train).numpy()
        val_acc = sca(y_val, y_pred_val).numpy()
        test_acc = sca(y_test, y_pred_test).numpy()
        accuracy = [ train_acc, val_acc, test_acc ]
        loss     = [ train_loss_unregularized, val_loss_unregularized, test_loss_unregularized ]
        return loss, accuracy
    else:
        return val_loss_unregularized


if __name__=="__main__":

    # ================== GOLBAL VARIABLES NEEDED TO RUN THE EXPERIMENT ======================

    #1] ENTER DATA DIRECTORIES
    base_dir  = "/content/drive/MyDrive/CIFAR_Dataset/CIFAR_10/Batch3_1+10K"
    train_dir = base_dir+'/cifar10_train.npz'
    val_dir   = base_dir+'/cifar10_val.npz'
    test_dir  = base_dir+'/cifar10_test.npz'
    # 2] ENTER DIRECTORY OF OPTUNA's TRAINED MODEL
    OPTUNA_MODEL_DIRECTORY = "/content/drive/MyDrive/CIFAR_Dataset/CIFAR_10/cnn_optuna.pickle"
    NUMBER_OF_DENSE_LAYERS = 3 # or 5; Includes the output layer.

    #3] GUROBI ENVIRONMENT WITH ACADEMIC LICENSE DETAILS
    # Note that acacdemic license is needed to solve large models.
    ENV = gp.Env( empty=True )
    ENV.setParam( 'WLSACCESSID', 'xxxxxxxxxxx' )   
    ENV.setParam( 'WLSSECRET', 'xxxxxxxxx' )    
    ENV.setParam( 'LICENSEID', xxxx )
    ENV.setParam( 'OutputFlag', 0 )      # To Turn-off Logs
    ENV.start()

    #4] RESULTS OUTPUT DIRECTORY
    full_output_directory   = "/content/drive/MyDrive/CIFAR_Dataset/CIFAR_10/result.xlsx"  # Second part is the name of the file. USE .xlsx in the end to save the result as excel file.

    #============================================  LOADING DATA  =====================================================
    x_train, x_val, x_test, y_train, y_val, y_test = data_loader( train_dir, val_dir, test_dir, model_type='cnn' )
    input_shape  = x_train.shape[1:]
    output_shape = 10
    ## **_______________________ FOR RESNET MODEL _____________________
    # CHANGE model_type to 'resnet50' 
    # Also, change "CNN" class to "CNN_ResNet50" with right inputs. such as in "loss_value" function.

    # LOADING OPTUNA TRAINED MODELS
    with open(OPTUNA_MODEL_DIRECTORY, "rb") as fout:
            list__ = pkl.load(fout)
    [ trainable_weights_list, full_weights_list, init_hyperparameters, weight_sets, grad_sets, optuna_time ] = list__

    #------------------------------------------------------------------------------------------------------------------
    # **WARNING : "model_" variable is used globally, so do not move the cell without editing the code.
    tf.keras.backend.clear_session()
    # Defining new dense layers in the end.
    # **(FOR ResNet50 modify accordingly)
    hidden_dense_layers = [64 for i in range(NUMBER_OF_DENSE_LAYERS-1)] 
    
    layer_info_ = layer_information( output_shape, hidden_dense_layers, include_flatten=False )
    model_= CNN( input_shape, output_shape).generate_model( layer_info_ )
    model_.build(input_shape) # Unless .build is called, gradient tape watch list will be empty
    model_.set_weights( full_weights_list )

    # GETTING INDEXES OF TRAINABLE WEIGHTS ONLY
    all_weights = model_.get_weights()
    trainable_weights = model_.trainable_weights
    index_set = []
    for idx in range(len(all_weights)):
        weight = all_weights[idx]
        var_name = model_.weights[idx].name.split(':')[0]
        if var_name in [t.name.split(':')[0] for t in trainable_weights]:
            index_set.append(idx)

    #------------------------------------------------------------------------------------------------------------------

    # GETTING OBJECTIVE COEFFICIENTS
    time1 = datetime.now()

    Validation_Coefficients = gradient_validation( batch_size = 128 )
    Validation_Coefficients = tf.concat( [tf.reshape(tensor, [-1]) for tensor in Validation_Coefficients], axis=0 ).numpy()

    time2 = datetime.now()
    coef_time = (time2 - time1).total_seconds()


    # RUNNING SEARCH FOR RANDOM NUMBER OF CONSTRAINTS
    percentage_of_submatrix = [ 0.01 ]
    num_hyperparams         = len(init_hyperparameters)

    trials = 1
    for trial in range(trials):
        for rows_ in percentage_of_submatrix:
            # GETTING OBJECTIVE COEFFICIENTS AND CONSTRAINT MATRIX
            time1 = datetime.now()
            constraint_matrix = random_constraints(rows_) # Set percentage of hessian to be used in the direction problem
            time2 = datetime.now()
            data_collection_time = (time2 - time1).total_seconds() + coef_time

            # GETTING DIRECTIONS FROM LINEAR PROGRAM
            Directions_ = Bilevel_Descent_Direction( Validation_Coefficients, constraint_matrix, 1e-4)
            linear_problem_runtime = Directions_[-1]

            Directions_ = np.array(Directions_[0])
            Directions_ = Directions_/np.linalg.norm(Directions_)

            time1 = datetime.now()

            layer_wise_shapes = [ val.shape for val in weight_sets[0] ]  # Only trainable layers
            layer_wise_params = [ val.flatten().shape for val in weight_sets[0] ]

            Weight_directions     = unflatten( Directions_[:-num_hyperparams] )
            Hyperparam_directions = Directions_[-num_hyperparams:]

            validation_loss_ = 1e10

            def minimize_function(t,Without_GSS=False):
                hyperparams_t   = [ tf.math.add( i,t*j ) for i,j in zip( init_hyperparameters, Hyperparam_directions  ) ]
                layer_weights_t = []
                for i,j in zip( trainable_weights_list, Weight_directions ):
                    new_weight = tf.math.add( tf.convert_to_tensor(i.numpy(), dtype = tf.float64),t*j )
                    layer_weights_t.append( new_weight )
                if Without_GSS:
                    loss, acc = loss_value( layer_weights_t, hyperparams_t, full_weights_list, Without_GSS )
                    return loss, acc
                else:
                    loss = loss_value( layer_weights_t, hyperparams_t, full_weights_list, Without_GSS )
                    return loss

            def interval_search( init_t = 0, step = 0.1 ):
                a = init_t
                b = init_t + step

                loss_a, loss_b = minimize_function(a), minimize_function(b)
                i=0
                while loss_b < loss_a:
                    i+=1
                    a,loss_a = b, loss_b
                    b = init_t + (2**i) * step
                    loss_b = minimize_function(b)
                if i==0:
                    return a,b
                elif i==1:
                    return init_t,b
                else:
                    return init_t + (2**(i-2)) * step, b

            def gss(tol=1e-5):
                
                a,b = interval_search()
                invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
                invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

                (a, b) = (min(a, b), max(a, b))
                h = b - a
                if h <= tol:
                    return (a, b)

                # Required steps to achieve tolerance
                n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

                c = a + invphi2 * h
                d = a + invphi * h
                yc = minimize_function(c)
                yd = minimize_function(d)

                for k in range(n - 1):
                    if yc < yd:  # yc > yd to find the maximum
                        b = d
                        d = c
                        yd = yc
                        h = invphi * h
                        c = a + invphi2 * h
                        yc = minimize_function(c)
                    else:
                        a = c
                        c = d
                        yc = yd
                        h = invphi * h
                        d = a + invphi * h
                        yd = minimize_function(d)

                if yc < yd:
                    return (a, d)
                else:
                    return (c, b)


            interval_ = gss(tol=1e-4 ) 
            time2 = datetime.now()

            optimal_t = (interval_[0] + interval_[1])/2
            loss, accuracy = minimize_function( optimal_t, Without_GSS=True) 
            init_loss, init_accuracy = minimize_function( 0, Without_GSS=True)
            print( "\nOptimal LOSS-> ", loss, "\nAccuracy -> ", accuracy, "\nt_star-> ", optimal_t )
            solution_search_runtime = (time2-time1).total_seconds()
            print("\n Total time taken for final improvement ::", solution_search_runtime, "\n\n")

            # ==================== Saving Results ================================
            DF = { "T_star"            : [ 0, optimal_t ],
                "Training_Loss"      : [ init_loss[0], loss[0] ],
                "Validation_Loss"    : [ init_loss[1], loss[1] ],
                "Testing_Loss"       : [ init_loss[2], loss[2] ],
                "Training_Accuracy"  : [ init_accuracy[0], accuracy[0] ],
                "Validation_Accuracy": [ init_accuracy[1], accuracy[1] ],
                "Testing_Accuracy"   : [ init_accuracy[2], accuracy[2] ],
                "Runtime"            :[ optuna_time.total_seconds(), data_collection_time + linear_problem_runtime + solution_search_runtime ]}
            
            DF = pd.DataFrame.from_dict(DF)
            DF.to_excel(full_output_directory)

            print("\n\n",DF)



    

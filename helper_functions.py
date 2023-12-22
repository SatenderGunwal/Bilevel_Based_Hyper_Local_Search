import numpy as np
import tensorflow as tf


# Enter training, validation and testing dataset directories for datasets.
def preprocess_image_input(input_images): # Only used when resent50 is selected as model_type below
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

def data_loader( train_dir, val_dir, test_dir, model_type ): # model_type : 'cnn' OR 'resnet50' ( Please specify only one of these in sttring format )

    train_dataset = np.load(train_dir)
    val_dataset   = np.load(val_dir)
    test_dataset  = np.load(test_dir)

    y_train = train_dataset['y_train'].astype("float32")
    y_val   = val_dataset['y_val'].astype("float32") 
    y_test  = test_dataset['y_test'].astype("float32") 

    x_train = train_dataset['x_train'].astype("float32")
    x_val   = val_dataset['x_val'].astype("float32")
    x_test  = test_dataset['x_test'].astype("float32") 

    if model_type == 'cnn':
        x_train, x_val, x_test = x_train/255, x_val/255, x_test/255
    elif model_type == 'resnet50':
        x_train = preprocess_image_input(x_train)
        x_val = preprocess_image_input(x_val)
        x_test = preprocess_image_input(x_test)
    elif model_type not in ['cnn', 'resnet50']:
        raise ValueError('Error: Please enter correct \'model_type\' variable value. Correct values are \'cnn\' or \'resnet50\' (strings).')
    return x_train, x_val, x_test, y_train, y_val, y_test

# This class is for the simple CNN model. For resnet50 (transfer learning) use below class.
class CNN: 

    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
    
    def generate_model(self, layer_info = None ):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(self.input_size, self.input_size, 3)))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.GlobalAveragePooling2D()) # add GlobalAveragePooling2D layer
        # model.add(tf.keras.layers.Flatten())

        if layer_info is not None:
            for layer in layer_info[:-1]:
                layer_params = layer['params']
                if layer['type'] == 'dense':
                    model.add(tf.keras.layers.Dense(**layer_params))
                    model.add(tf.keras.layers.BatchNormalization())
            for layer in layer_info[-1:]:
                layer_params = layer['params']
                if layer['type'] == 'dense':
                    model.add(tf.keras.layers.Dense(**layer_params))

        # Calculate the number of trainable parameters in the model
        trainable_count = sum(tf.keras.backend.count_params(weights) for weights in model.trainable_weights)
        print(f"Trainable parameters: {trainable_count:,}")

        return model
    
# Be sure to replace CNN with CNN_ResNet50 in the optuna_training.py file if using resnet.
class CNN_ResNet50: 
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def generate_model(self, layer_info = None ):
        UpSampling = 224/self.input_shape[0]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        model.add( tf.keras.layers.UpSampling2D(size=(UpSampling,UpSampling)))
        model.add(tf.keras.applications.resnet50.ResNet50(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3),  pooling = 'avg'))
        if layer_info is not None:
            for layer in layer_info[:-1]:
                layer_params = layer['params']
                if layer['type'] == 'dense':
                    model.add(tf.keras.layers.Dense(**layer_params))
                    model.add(tf.keras.layers.BatchNormalization())
            for layer in layer_info[-1:]:
                layer_params = layer['params']
                if layer['type'] == 'dense':
                    model.add(tf.keras.layers.Dense(**layer_params))
        model.layers[1].trainable = False
        # Calculate the number of trainable parameters in the model
        trainable_count = sum(tf.keras.backend.count_params(weights) for weights in model.trainable_weights)
        print(f"Trainable parameters: {trainable_count:,}")
        return model
    

# ========= OTHER FUNCTIONS NEEDED FOR HYPER LOCAL TUNING ==================

def layer_information( output_classes, dense_units, dense_kernel_regularizers=None, output_kernel_regularizer=None, include_flatten=True ):
    layers = []
    if include_flatten: layers.append({'type': 'flatten', 'params': {}})
    if dense_kernel_regularizers != None:
        for units,lamda in zip(dense_units,dense_kernel_regularizers):
            layers.append( {'type': 'dense', 'params': {'units': units, 'activation': 'relu', 'kernel_regularizer':tf.keras.regularizers.l2(lamda)}} )
    else:
        for units in dense_units:
            layers.append( {'type': 'dense', 'params': {'units': units, 'activation': 'relu'}} )
    if output_kernel_regularizer == None:
        layers.append( {'type': 'dense', 'params': {'units': output_classes, 'activation': 'softmax'}} )
    else: layers.append( {'type': 'dense', 'params': {'units': output_classes, 'activation': 'softmax', 'kernel_regularizer':tf.keras.regularizers.l2(output_kernel_regularizer)}} )

    return layers

# # Extra functions required for operations during hessian approximation
# def compute_outer_product(tensors1, tensors2): # USES CPU RAM

#     arr1, arr2 = tensors1[0].reshape(-1), tensors2[0].reshape(-1)
#     for tensor1, tensor2 in zip( tensors1[1:], tensors2[1:] ):
#         arr1 = np.concatenate((arr1, tensor1), axis=None)
#         arr2 = np.concatenate((arr2, tensor2), axis=None)
#     return np.outer(arr1, arr2)

# def compute_inner_product( tensors1, tensors2 ):

#     flattened_vector1 = [tf.reshape(t, [-1]) for t in tensors1]
#     flattened_vector1 = tf.concat(flattened_vector1, axis=-1)

#     flattened_vector2 = [tf.reshape(t, [-1]) for t in tensors2]
#     flattened_vector2 = tf.concat(flattened_vector2, axis=-1)

#     flattened_shape = flattened_vector1.shape.as_list()

#     matrix1 = tf.linalg.LinearOperatorFullMatrix(
#         tf.reshape(flattened_vector1, [flattened_shape[0], 1])
#     )
#     matrix2 = tf.linalg.LinearOperatorFullMatrix(
#         tf.reshape(flattened_vector2, [1,flattened_shape[0]])
#     )
#     return tf.linalg.matmul(matrix2, matrix1).to_dense()



# using separate loss expression for regularization loss
def part_loss( regularized_weight_vars, Regularization_Parameters ):

    total_loss = tf.constant(0, dtype = tf.float32)
    for weight, lamda in zip( regularized_weight_vars, Regularization_Parameters ): 
        regularization  = lamda * tf.reduce_sum(tf.square(weight))
        regularization /= tf.cast(tf.size(weight), dtype=tf.float32)
        total_loss     += regularization

    return total_loss

def remaining_approximation( trainable_weights, Regularization_Parameters ):

    # Making the Hyperparameters tf variables (if not)
    for idx in range(len(Regularization_Parameters)):
        Regularization_Parameters[idx] = tf.Variable( Regularization_Parameters[idx] )
    
    # Getting indexes of variables being regularized based on dense layer structure. ( NOTE_: This only works if only dense layers are regularized )
    Weights_Index_ToRegularize = []
    for idx in range(len(trainable_weights)):
        if len(trainable_weights[idx].shape) == 2:
            Weights_Index_ToRegularize.append(idx)

    All_Weights = [ tf.Variable(weight) for weight in trainable_weights ]
    weight_vars = [ All_Weights[idx] for idx in Weights_Index_ToRegularize]
    del(trainable_weights)

    with tf.GradientTape(persistent = True) as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss_ = part_loss( weight_vars, Regularization_Parameters )
        grads = inner_tape.gradient(loss_, All_Weights, unconnected_gradients="zero")

    second_derivatives = []
    for g in grads:
        jacob = outer_tape.jacobian( g, Regularization_Parameters, unconnected_gradients ='zero') 
        second_derivatives.append( tf.convert_to_tensor(jacob) )

    # Combining the second-derivative entries
    for idx in range(len(second_derivatives)):
        tensor = second_derivatives[idx]
        if len(tensor.shape) >= 3:
            second_derivatives[idx] = tf.reshape(tensor, (tensor.shape[0], -1))
    # Concatenate matrix tensors row-wise
    second_derivatives = tf.concat(second_derivatives, axis=1)
    return second_derivatives
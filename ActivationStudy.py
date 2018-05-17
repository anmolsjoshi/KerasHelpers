"""
author: joshi
"""

from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import categorical_crossentropy, binary_crossentropy
import deepdish as dd
import os


class GradientActivationStore(Callback):

    """
    Keras Callback to save Gradients and Activations on validation set during training process.

    This callback saves activations and gradients on the first batch after training, for initial parameters, and then
    saves parameters at the end of each epoch.

    The gradients and activation are saved as a dictionary in .h5 format.

    For example - parameters['epoch0']['dense_1'] contains the activations of the first layer at the start of training
    """

    def __init__(self, DIR, filename, num_classes, record_every=None, only_weights=False):
        super(GradientActivationStore, self).__init__()

        """
        Input:
            - DIR: directory where activations and gradients are saved
            - filename: name of file, preferably name of experiment for better recording
            - num_classes: number of classes
            - record_every: int, save parameters at what interval of epochs
            - only_weights: boolean, True to save gradients for weights only
        """

        # Creates directory with subfolders to save gradients and activations
        if not os.path.exists(DIR):
            os.makedirs(DIR)
            os.makedirs(os.path.join(DIR, 'activations'))
            os.makedirs(os.path.join(DIR, 'gradients'))

        self.DIR = DIR
        self.filename = filename

        # Placeholder is required to calculate gradients and activations
        self.num_classes = num_classes
        self.y_true = K.placeholder(shape=[None, num_classes])

        self.record_every = record_every
        self.only_weights = only_weights

        # Initialize empty dictionaries
        self.gradients = {}
        self.activations = {}

        # on_train_begin does not allow access to the validation set. Functions are called on first batch of first epoch
        # Once complete, the flag is turned to False to prevent further use.
        self.initial_flag = True

    def on_train_begin(self, logs=None):

        """
        This function extracts the names of the layers, weights and biases. These are later used by dict_create_append

        only_weights flag is used to save weights and/or biases.
        """

        if self.only_weights:
            self.weight_names = [weight.name.split(':')[0].replace('/', '_')
                                 for weight in self.model.trainable_weights if 'kernel' in weight.name]
        else:
            self.weight_names = [weight.name.split(':')[0].replace('/', '_') for weight in self.model.trainable_weights]

        self.activation_names = [layer.name for layer in self.model.layers[1:]]

    def dict_create_append(self, dictionary, epoch, names, values):

        """
        Input:
            - dictionary: Dictionary to add keys and parameters
            - epoch: Epoch number, int
            - names: output of layer names/parameter names from on_train_begin
            - values: gradients/activations to be saved to dictionary

        Returns:
             - dictionary: dictionary with added key of epoch number, containing keys of layer names/parameter names
                            containing parameter values
        """

        dictionary['epoch'+str(epoch)] = {}
        for i, name in enumerate(names):
            dictionary['epoch'+str(epoch)][name] = values[i]
        return dictionary

    def get_gradients(self, model):

        """
        This function outputs a function to calculate the gradients based on the loss function, current weights/biases

        Input:
            - model: model that is training.

        Returns:
             - func: a function that uses input of features and true labels to calculate loss and hence gradients
        """

        model_weights = model.trainable_weights

        if self.only_weights:
            weights = [weight for weight in model_weights if 'kernel' in weight.name]
        else:
            weights = [weight for weight in model_weights]

        if self.num_classes > 1:
            loss = K.mean(categorical_crossentropy(self.y_true, model.output))
        else:
            loss = K.mean(binary_crossentropy(self.y_true, model.output))

        func = K.function([model.input, self.y_true], K.gradients(loss, weights))
        return func

    def get_activations(self, model):

        """
        This function outputs a function to calculate the gradients based on the loss function, current weights/biases

        Input:
            - model: model that is training.

        Returns:
            - func: a function that uses input of features outputs activations based on current parameters
        """

        func = K.function([model.input], [layer.output for layer in model.layers[1:]])  # evaluation function
        return func

    def call_grads_acts(self, epoch):

        """
        Function that calls get_grad, get_activations, dict_create_append to save gradients and activations on
        validation set when called.
        """
        # Gradients
        get_grad = self.get_gradients(self.model)
        inputs = [self.validation_data[0], self.validation_data[1]]
        grads = get_grad(inputs)
        self.gradients = self.dict_create_append(self.gradients, epoch, self.weight_names, grads)

        # Activations
        get_act = self.get_activations(self.model)
        acts = get_act([self.validation_data[0]])
        self.activations = self.dict_create_append(self.activations, epoch, self.activation_names, acts)

    def on_batch_end(self, batch, logs=None):

        """
        This function is only used to capture the parameters at the start of training
        """

        if self.initial_flag == True:
            self.call_grads_acts(epoch=0)
        self.initial_flag = False

    def on_epoch_end(self, epoch, logs=None):

        """
        This function calls call_grads_acts at the record_every interval
        """

        if epoch%self.record_every == 0:
            self.call_grads_acts(epoch=epoch+1)

    def on_train_end(self, logs=None):

        """
        Saves gradients and activations of validation set to disk
        """

        dd.io.save(os.path.join(self.DIR, 'gradients', self.filename + '-gradients.h5'), self.gradients)
        dd.io.save(os.path.join(self.DIR, 'activations', self.filename + '-activations.h5'), self.activations)
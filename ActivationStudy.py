import os
import deepdish as dd
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import categorical_crossentropy, binary_crossentropy

'''
author: joshi
'''

class GradientActivationStore(Callback):

    def __init__(self, DIR, num_classes, record_every=None, only_weights=False):
        super(GradientActivationStore, self).__init__()

        '''
        num_classes: int, number of classes for classification
        loss_function: str, binary_crossentropy, categorical_crossentropy 
        DIR: str, directory to save activations and gradients history   
        save_every: int, how many epochs to save data at    
        
        '''

        if not os.path.exists(DIR):
            os.makedirs(DIR)
            os.makedirs(os.path.join(DIR, 'activations'))
            os.makedirs(os.path.join(DIR, 'gradients'))

        self.filename = DIR
        self.num_classes = num_classes
        self.y_true = K.placeholder(shape=[None, num_classes])

        self.record_every = record_every
        self.only_weights = only_weights

        self.gradients = {}
        self.activations = {}


    def dict_create_append(self, dictionary, epoch, names, values):
        dictionary['epoch'+str(epoch)] = {}
        for i, name in enumerate(names):
            dictionary['epoch'+str(epoch)][name] = values[i]
        return dictionary

    def get_gradients(self, model):

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
        func = K.function([model.input], [layer.output for layer in model.layers[1:]])  # evaluation function
        return func

    def call_grads_acts(self, epoch):
        # Gradients
        get_grad = self.get_gradients(self.model)
        inputs = [self.validation_data[0], self.validation_data[1]]
        grads = get_grad(inputs)
        self.gradients = self.dict_create_append(self.gradients, epoch, self.weight_names, grads)

        # Activations
        get_act = self.get_activations(self.model)
        acts = get_act([self.validation_data[0]])
        self.activations = self.dict_create_append(self.activations, epoch, self.activation_names, acts)

    def on_train_begin(self, logs=None):

        if self.only_weights:
            self.weight_names = [weight.name.split(':')[0].replace('/', '_')
                                 for weight in self.model.trainable_weights if 'kernel' in weight.name]
        else:
            self.weight_names = [weight.name.split(':')[0].replace('/', '_') for weight in self.model.trainable_weights]

        self.activation_names = [layer.name for layer in self.model.layers[1:]]

        #self.call_grads_acts(epoch=0)

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.record_every == 0:
            self.call_grads_acts(epoch=epoch)

    def on_train_end(self, logs=None):
        #dd.io.save('gradients.h5', self.gradients)
        #print('COMPLETE')
        dd.io.save('activations.h5', self.activations)
        print('COMPLETE')
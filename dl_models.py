# This module is used specifically for dl purposes.
from Utils.utils import *

import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, Input
from keras.layers import Lambda, Flatten, RepeatVector, Permute, Multiply
from keras.layers import LSTM, GRU, Bidirectional, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Concatenate

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop, Adam

from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import sys

import itertools

class ModelExperiments():
    def __init__(self, network_parameters, indexes, model_dir="./models/"):
        """ 
        This class is the one in charge of getting the set of configurations for the experiments, generating the different
        possible model configurations, compile and train the different models.

        Appart from those parameters, other such as the optimizer, and the patience, can as well be customized.
        """
        # Class variables
        self.model_list = []
        self.experiment_names = []
        self.model_callbacks = []
        
        self.model_dir = model_dir
        check_dir(self.model_dir)
        
        # Generate the list of experiment combinations and their names
        self.experiments_generator = ExperimentsGenerator(network_parameters, indexes)     
        self.network_parameters = self.experiments_generator.combinations
        self.experiment_names  = self.experiments_generator.experiment_names
        
        # Generate the models.
        self.model_generator = ModelGenerator()
        
    def compile_models(self, verbose=False, optimizer="adam"):
        """
        This function will create the models with the parameters passed to the network. It returns the models as well as their 
        respective names.
        
        Parameters:
            verbose: Defines if the summary of the models is shown or not. Set to False.
        """
        for net_p in self.network_parameters:
            m = self.model_generator.create_model(net_p)
            
            m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
            self.model_list.append(m)
    
            if verbose:
                m.summary()
        
        return self.model_list, self.experiment_names
            
    def train_models(self, x, y, validation=None, retrain_models=False):
        """
        This method trains all the compiled models of the experiments. It receives as input the training and validation data,
        otherwise.
        """
        for i, net_model in enumerate(self.model_list):
            # Set a name for the model based on the tweaked parameters
            p = self.network_parameters[i]
            name = self.experiment_names[i]
            model_path = self.model_dir + name

            if not retrain_models:
                # If the model exists, don't compute it again.
                if os.path.isfile(model_path):
                    continue

            print("\n\n********************************************\n")    
            print(name)
            callbacks = self._create_callbacks(name, i)
            # Fit the model and extract its data
            print("Callbacks are the following")
            print(callbacks)
            try:
                history = net_model.fit(x, y,
                                        # Uses the 25% of the train as validation. This validation can be parametrizable too.
                                        validation_split=0.25,
                                        # Even with early-stopping this could be good to parameterize. 
                                        epochs=self.network_parameters[i].get("epochs", 20),                        
                                        batch_size=self.network_parameters[i].get("batch_size", None),
                                        callbacks=callbacks,
                                        # This will ponderate the classes. Use with unbalanced datasets.
                                        # Adding it as a parameter might be a good idea.
                                        #class_weight={0: 0.11, 1: 0.89} 
                                       )
            except Exception as e:
                print("Could not Train Model")
                print(e)
                continue


        # To free memory from the gpu
        # This is used for long collections of experiments.
        from keras import backend as K
        K.clear_session()

        
    def _create_callbacks(self, name, i):
        """
        This function generates the callback for each training of the network. TensorBoard is also included
        in these callbacks.
        
        The tensorboard logs will be stored in a folder with the configuration of the experiment as name.
        
        parameters:
            name: the name of the training for the logs of the tensorboard.
            i: The position of the experiment in the posible configurations.
        """
        # Directory where the tensorboard will be stored.
        log_dir = self.model_dir + 'logs/' + name
        
        # Define the TB callback.
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

        # Generate the dir if it does not exist.
        check_dir(log_dir)
        patience = self.network_parameters[i].get("patience", 4)
        
        # Collection of callbacks for that experiment.
        callbacks = [
            ReduceLROnPlateau(patience=patience/2),
            EarlyStopping(patience=patience), 
            tensorboard
        ]
        
        return callbacks
        
        
class ExperimentsGenerator():
    def __init__(self, parameters, indexes):
        """
        This class will generate the different experiments with their combinations
        as well as their names
        """
                
        try:
            assert len(parameters) == len(indexes)
            assert len(indexes) > 0
        except AssertionError:
            print('Number of parameters and indexes must be the same and bigger than zero.')
            print("Parameters: {}\nIndexes: {}".format(len(parameters[0]), len(indexes)))
            return None

        self.parameters = parameters
        self.indexes = indexes
        self.combinations = self._create_combinations()
        self.experiment_names = self._create_names()

        
    def _create_combinations(self):
        """
        This function is in charge of combining the parameters. This way we can perform a grid search automatically.

        Inputs: The array of different parameters, and the indexes for them.

        Outputs: The dictionary of different configurations made.
        """
        combinations = list(itertools.product(*self.parameters))
        param_combinations = [{k:v for k, v in zip(self.indexes, combination)}  for combination in combinations]
                
        return param_combinations
    
    def _create_names(self):
        """
        This function is in charge of generating the names of the experiments. 
        These names will be based on the parameters each configuration has.
        """
        network_names = []
        try:
            assert len(self.combinations) > 0
        except AssertionError:
            print("In order to get the names, the combinations mus be computed first")
            return None
       
        for p in self.combinations:
            name = ""
            for k, v in p.items():
                name += "{}_{}_".format(k, v)
            else:
                name = name[:-1]

             # This is used to clean the names of possible keras classes as well as punctuations.    
            name = name.replace(" ", "").replace("[", "").replace("]", "").replace(",", "-").replace("'", "").replace(">", "").replace("<classkeras.layers.recurrent.", "")
            
            network_names.append(name)
            
        return network_names
    
    
class ModelGenerator():
    # This will be used as a class attribute. Because loading embeddings is a really costly process but we want this 
    # class to be a factory of models we cant have normal class methods.
    w2v_model = None
    
    def __init__(self):
        """
        # This class will generate the different models depending on the configuration of the experimets passed.
        # It will do so by stacking the layers in the *create_model* configuration
        """
        pass
    
    @classmethod
    def create_model(cls, params):
        """
        This method creates a network model with the parameters given.

        Returns the uncompiled model.
        """
        
        inputs = Input(name='inputs',shape=(params["input_length"],))

        z = cls._add_embeddings(inputs, params)

        z = cls._add_cnn(z, params)

        z = cls._add_rnn(z, params)

        z = cls._add_dnn(z, params)

        # This layer uses a sigmoidal activation for 1 class or a softmax in case of a multi-classification problem.
        outputs = Dense(params.get("output_length", 1),
                        activation='sigmoid' if params.get("output_length") == 1 else 'softmax',
                        name='output_layer')(z)

        net_model = Model(inputs=inputs,outputs=outputs)

        return net_model
    
    @classmethod
    def _add_embeddings(cls, z, params): 
        """
        This method adds embeddings to the network.

        Returns the net with the embeddings added.

        Parameters:
            z: The model up to now.
            parameters: These are the dictionary of parameters of the network. This may contain the following ones:
                load: Defines if the embeddings are going to be loaded. Defaults to False.
                size: Defines the size of the embedding vector. Must match size if the embeddings are loaded. Defaults to 300.
                trainable: Defines if the embeddings can be trainable. Defaults to the oposite of load.
                vocab_size: Defines the size of the vocabulary. Defaults to 5000.
                input_length: Defines the length of the sequences it gets as input. Must be defined.
                embedding_matrix: Defines the weights used if loading the embeddings.
        """
        # Get the parameters needed for the embeddings
        load = params.get('load_emb', False)
        size = params.get('emb_size', 300)
        trainable = params.get('trainable', not load) # If not loading, default is to train the embeddings
        vocab_size = params.get('vocabulary_length', 5000)
        input_length = params.get('input_length')
        embedding_matrix = params.get('embedding_matrix', None)
        
                
        if load:
            if embedding_matrix is None:
                print("No embeddings matrix added as a parameter. Using Randomly initialized one instead")
                load = False
        
        if load:    
            z = Embedding(vocab_size, size, input_length=input_length, weights=[embedding_matrix], trainable=trainable)(z)
        else:
            z = Embedding(vocab_size, size, input_length=input_length)(z)

        return z

    @classmethod
    def _add_cnn(cls, z, params):
        """
        This method adds the unidimensional CNN layers to the network. After each cnn, a MaxPooling is applied.

        Returns the model with the added CNN layers.

        Parameters:
            z: The model up to now.
            parameters: These are the dictionary of parameters of the network. This may contain the following ones:
                size: Defines the number of filters and the number of layers.
                filter_sizes: Defines the size of the filters used, as well as the number of them.
                flatten: Defines if a flatten layer is added at the end of the cnn ones.
        """
        # Get the parameters needed for the cnns
        size = params.get("cnn_size", None)
        filter_sizes = params.get("cnn_filter", 3)
        flatten = [None] == params.get("rnn_size", [None])  # This can be changed if the topology (cnn followed by rnn) changes.        
            
        conv_blocks = []
        for filter_size in filter_sizes:
            if filter_size is None:
                return z
            conv = None
            for i, cnn_layer in enumerate(size):
                if cnn_layer is None:
                    return z
                conv = Conv1D(cnn_layer, filter_size, padding='valid', activation='relu', strides=1)(z if conv is None else conv)
                conv = MaxPooling1D(pool_size=filter_size)(conv)        

            if flatten:
                conv = Flatten()(conv)
            conv_blocks.append(conv)         

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        return z

    @classmethod
    def _add_rnn(cls, z, params):
        """
        This method adds the RNN layers to the network. It also adds an attention layer if intended.

        Returns the net with the RNN & Attention added.

        parameters:
            z: The model up to now.
            params: These are the dictionary of parameters of the network. This may contain the following ones:
                size: The vector of sizes of the desired rnn. 
                bidirectional: Defines if the rnn layers are bidirectional ones. This will affect to all layers, so caution is advised.
                cell_type: Defines the type of the recurrent cell used (GRU OR LSTM)
                attention: Defines if attention is used as a pooling method.
        """
        # Get the parameters needed for the RNNs
        size = params.get("rnn_size", None)
        bidirectional = params.get("bidirectional", False)
        cell_type = params.get("cell_type", GRU)
        attention = params.get("attention", False)
        
        for i, rsz in enumerate(size):
            if rsz is None:
                return z

            if not bidirectional:
                if i < len(size) - 1:
                    z = cell_type(rsz, return_sequences=True)(z)
                else:
                    z = cell_type(rsz, return_sequences=attention)(z)

            else:
                if i < len(size) - 1:
                    z = Bidirectional(cell_type(rsz, return_sequences=True))(z)
                else:
                    z = Bidirectional(cell_type(rsz, return_sequences=attention))(z)

            if attention:
                z = cls._add_attention(z)

        return z

    @classmethod
    def _add_dnn(cls, z, params):
        """
        This method adds the DNN layers to the network.

        Returns the net with the DNN layers added.

        parameters:
            z: The model up to now.
            params: These are the dictionary of parameters of the network. This may contain the following ones:
                size: The vector of sizes of the desired dnn.
                dropout: The dropout used.
                activation: The activation of the cells. Set to relu.
        """
        
        size = params.get("dnn_size", None)
        dropout = params.get("dropout", 0.5)
        activation= params.get("activation", "relu")
        for fsz in size:
            if fsz is None:
                return z

            z = Dense(fsz, activation=activation)(z)
            z = Dropout(dropout)(z)

        return z

    @classmethod
    def _add_attention(cls, z):
        """
        This method applies an attention layer to the rnns.

        Returns the model with the attention layer.
        """

        size =  K.int_shape(z)[-1]
        attention = BatchNormalization()(z)
        attention = Dense(1, activation='tanh')(attention)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(size)(attention)
        attention = Permute([2, 1])(attention)

        z = Multiply()([z, attention])
        z = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(size,))(z)

        return z
    
    @staticmethod
    def load_W2V_model(path, binary):
        """
        This function is used to load a w2v model into the class.
        
        This function is only needed if parameters of loading embeddings are set to true.
        """
        model = KeyedVectors.load_word2vec_format(path, binary=binary)
        print("Loaded W2V model")
        
        return model
    
    @staticmethod
    def generate_embedding_matrix(word_index, max_words, model):
        # We initialize the embd matrix as random
        # embedding_matrix = np.zeros((max_words, model.vector_size), dtype=np.float32)
        embedding_matrix = np.random.rand(max_words, model.vector_size)
        hit = 0
        for word, i in word_index.items():
            if i >= max_words:
                break
            if word in model:
                # words not found in embedding index will be randomly initialized.
                embedding_matrix[i] = model[word]
                hit += 1
        print("Hits: {}\nTotal: {}".format(hit, i))

        return embedding_matrix
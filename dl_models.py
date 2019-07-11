# This module is used specifically for dl purposes.
from utils import *

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

import itertools

class ModelExperiments():
    def __init__(self, network_parameters, indexes, retrain_models=False, model_dir="./models"):
        """ 
        This class is the one in charge of getting the set of configurations for the experiments, generating the different
        possible model configurations, compile and train the different models.

        Appart from those parameters, other such as the optimizer, and the patience, can as well be customized.
        """
        # Class variables
        self.model_list = []
        self.experiment_names = []
        self.model_callbacks = []
        self.retrain_models = retrain_models
        
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
            
    def train_models(x, y):
        """
        This method trains all the compiled models of the experiments. It receives as input the training and validation data,
        othersi
        """
        for i, net_model in enumerate(self.model_list):
            # Set a name for the model based on the tweaked parameters
            p = self.network_parameters[i]
            name = self.experiment_names[i]
            model_path = self.models_dir + name

            if not self.retrain_models:
                # If the model exists, don't compute it again.
                if os.path.isfile(model_path):
                    continue

            print("\n\n********************************************\n")    
            print(name)
            callbacks = self._create_callbacks(name)
            # Fit the model and extract its data
            try:
                history = net_model.fit(train_x_token, train_y,
                                        validation_split=0.25,
                                        epochs=20, 
                                        batch_size=network_parameters[i]["batch_size"], 
                                        callbacks=callbacks,
                                        class_weight={0: 0.11, 1: 0.89} # Classes are weighted proportionally.
                                       )
            except Exception as e:
                print("Could not Train Model")
                print(Exception)
                continue

            # And save the model
            net_model.save(model_path)

        # To free memory from the gpu
        from keras import backend as K
        K.clear_session()

        
    def _create_callbacks(self, name):
        """
        This function generates the callback for each training of the network. TensorBoard is also included
        in these callbacks.
        
        The tensorboard logs will be stored in a folder with the configuration of the experiment as name.
        """
        # Directory where the tensorboard will be stored.
        log_dir = self.model_dir + 'logs/' + name
        
        # Define the TB callback.
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

        # Generate the dir if it does not exist.
        check_dir(log_dir)
        
        # Collection of callbacks for that experiment.
        callbacks = [
            ReduceLROnPlateau(patience=2),# Can change the patience used. Maybe add as parameter.
            EarlyStopping(patience=4),    # Can change the patience used. Maybe add as parameter.
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
                
            name = name.replace(" ", "").replace("[", "").replace("]", "").replace(",", "-").split(".")[-1].replace("'", "").replace(">", "")
            
            network_names.append(name)
    
        return network_names
    
        """
        network_names = []
        for i, p in enumerate(network_parameters):
            name = "load_emb_{}_num_classes_{}_emb_size_{}_trainable_emb_{}_cnn_size_{}_cnn_filter_{}_pool_rnn_size_{}_cell_type_{}_bidirectional_{}_attention_{}_dropout_{}_dnn_size_{}_batch_size_{}".format(
              p["load_emb"], p["num_classes"], p["emb_size"], p["trainable_emb"], p["cnn_size"], p["cnn_filter"],
              p["rnn_size"], str(p["cell_type"]).split(".")[-1].replace("'", "").replace(">", ""), p["bidirectional"],
              p["attention"], p["dropout"], p["dnn_size"], p["batch_size"])
            
            name = name.replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")
            self.experiment_names.append(name)
    
        return self.experiment_names
        """
    
class ModelGenerator():
    def __init__(self):
        """
        # This class will generate the different models depending on the configuration of the experimets passed.
        # It will do so by stacking the layers in the *create_model* configuration
        """
        pass
    
    def create_model(self, params):
        """
        This method creates a network model with the parameters given.

        Returns the uncompiled model.
        """
        inputs = Input(name='inputs',shape=(params["input_length"],))

        z = self._add_embeddings(inputs, params["load_emb"], params["emb_size"], params["trainable_emb"],
                                 params["vocabulary_length"] ,params["input_length"], params["embedding_matrix"])

        z = self._add_cnn(z, params["cnn_size"], params["cnn_filter"], [None] == params["rnn_size"])

        z = self._add_rnn(z, params["rnn_size"], params["bidirectional"], params["cell_type"], params["attention"])

        z = self._add_dnn(z, params["dnn_size"], params["dropout"])

        outputs = Dense(params["num_classes"], activation='sigmoid', name='output_layer')(z)

        net_model = Model(inputs=inputs,outputs=outputs)

        return net_model
    
    @staticmethod
    def _add_embeddings(z, load, size, trainable, vocab_size, input_length, embedding_matrix=None): 
        """
        This method adds embeddings to the network.

        Returns the net with the embeddings added.

        Parameters:
            z: The model up to now.
            load: Defines if the embeddings are going to be loaded.
            size: Defines the size of the embedding vector. Must match size if the embeddings are loaded.
            trainable: Defines if the embeddings can be trainable. Set to True if load is false.
            vocab_size: Defines the size of the vocabulary.
            input_length: Defines the length of the sequences it gets as input.
            embedding_matrix: Defines the weights used in case of load==true.
        """
        if load:
            z = Embedding(vocab_size, size, input_length=input_length, weights=[embedding_matrix], trainable=trainable)(z)
        else:
            z = Embedding(vocab_size, size, input_length=input_length)(z)

        return z

    @staticmethod
    def _add_cnn(z, size, filter_sizes, flatten):
        """
        This method adds the unidimensional CNN layers to the network. After each cnn, a MaxPooling is applied.

        Returns the model with the added CNN layers.

        Parameters:
            z: The model up to now.
            size: The vector of sizes of the desired cnn. 
            filter_sizes: The vector of sizes of the desired filters. Multiple filters can be applied at the same time. This will
            generate diverse branches (one per filter size) which will concatenate after the convolutions.
            flatten: Defines if the output data must be flattened or not.
        """
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

    @staticmethod
    def _add_rnn(z, size, bidirectional, cell_type, attention):
        """
        This method adds the RNN layers to the network. It also adds an attention layer if intended.

        Returns the net with the RNN & Attention added.

        parameters:
            z: The model up to now.
            size: The vector of sizes of the desired rnn. 
            bidirectional: Defines if the rnn layers are bidirectional ones. This will affect to all layers, so caution is advised.
            cell_type: Defines the type of the recurrent cell used (GRU OR LSTM)
            attention: Defines if attention is used as a pooling method.
        """
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
                z = _add_attention(z)

        return z

    @staticmethod
    def _add_dnn(z, size, dropout, activation="relu"):
        """
        This method adds the DNN layers to the network.

        Returns the net with the DNN layers added.

        parameters:
            z: The model up to now.
            size: The vector of sizes of the desired dnn.
            dropout: The dropout used.
            activation: The activation of the cells. Set to relu.
        """
        for fsz in size:
            if fsz is None:
                return z

            z = Dense(fsz, activation=activation)(z)
            z = Dropout(dropout)(z)

        return z

    @staticmethod
    def _add_attention(z):
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
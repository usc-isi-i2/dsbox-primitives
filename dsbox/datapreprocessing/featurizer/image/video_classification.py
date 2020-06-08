import logging
import importlib
import numpy as np
import sys
import typing
import time
import copy
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import container

# lazy init
# import keras
# from tensorflow.keras.engine.sequential import Sequential

from . import config

# Input should from dataframe_to_tensor primitive
Inputs = container.DataFrame
Outputs = container.DataFrame  # results


class LSTMHyperparams(hyperparams.Hyperparams):
    LSTM_units = hyperparams.UniformInt(
        default=2048,
        lower=1,
        upper=pow(2, 31),
        description="Positive integer, dimensionality of the output space of LSTM model",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    verbose = hyperparams.UniformInt(
        default=0,
        lower=0,
        upper=3,
        description="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    batch_size = hyperparams.UniformInt(
        default=32,
        lower=1,
        upper=10000,
        description="The batch size for RNN training",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    epochs = hyperparams.UniformInt(
        default=1000,
        lower=1,
        upper=sys.maxsize,
        description="epochs to do on fit process",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter", ]
    )
    shuffle = hyperparams.Hyperparameter[bool](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=True,
        description='Shuffle minibatches in each epoch of training (fit).'
    )
    loss_threshold = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Threshold of loss value to early stop training (fit).'
    )
    weight_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-6,
        description='Weight decay (L2 regularization) used during training (fit).'
    )
    learning_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'
    )
    dropout_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.5,
        description='Learning rate used during training (fit).'
    )
    validate_data_percent = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.2,
        description='The perncentage of the data used for validation purpose during train(fit).'
    )
    optimizer_type = hyperparams.Enumeration[str](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values=['adam', 'sgd'],
        default='adam',
        description='Type of optimizer used during training (fit).'
    )
    momentum = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.9,
        description='Momentum used during training (fit), only for optimizer_type sgd.'
    )


class Params(params.Params):
    target_column_name: str
    class_name_to_number: typing.Dict[str, int]
    keras_model: typing.Dict  # tensorflow.keras.engine.Sequential, cannot use because of lazy init
    feature_shape: typing.List[int]
    input_feature_column_name: str


class LSTM(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, LSTMHyperparams]):
    """
    video classification primitive that use lstm RNN network
    """

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-video-classification-lstm',
        'version': config.VERSION,
        'name': "DSBox Video Classification LSTM",
        'description': 'Find the corresponding object position from given images(tensors)',
        'python_path': 'd3m.primitives.classification.lstm.DSBOX',
        'primitive_family': "CLASSIFICATION",
        'algorithm_types': ["DEEP_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'lstm'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            'uris': [config.REPOSITORY]
        },
        # The same path the primitive is registered with entry points in setup.py.
        'installation': [config.INSTALLATION],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: LSTMHyperparams, volumes: typing.Union[typing.Dict[str, str], None] = None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = 0
        self._inited = False
        self._model = None  # this used to store the real of the model
        self._training_inputs = None
        self._training_outputs = None
        self._epochs = self.hyperparams['epochs']
        self._batch_size = self.hyperparams['batch_size']
        self._loss_threshold = self.hyperparams['loss_threshold']
        self._validate_data_percent = self.hyperparams['validate_data_percent']
        self._shuffle = self.hyperparams['shuffle']
        self._verbose_mode = self.hyperparams['verbose']
        self._learning_rate = self.hyperparams['learning_rate']
        self._weight_decay = self.hyperparams['weight_decay']
        self._LSTM_units = self.hyperparams['LSTM_units']
        self._dropout_rate = self.hyperparams['dropout_rate']
        self._optimizer_type = self.hyperparams['optimizer_type']
        self._momentum = self.hyperparams['momentum']
        self._metrics = []
        self._class_name_to_number = {}
        self._input_feature_column_name = ""
        self.logger = logging.getLogger(__name__)

    def get_params(self) -> Params:
        param = Params(
            keras_model=self._model.get_config(),
            class_name_to_number=self._class_name_to_number,
            target_column_name=self._target_column_name,
            feature_shape=self._feature_shape,
            input_feature_column_name=self._input_feature_column_name
        )
        return param

    def set_params(self, *, params: Params) -> None:
        from tensorflow.keras import Sequential
        # self._model = self._lazy_init_lstm()
        config_lstm = params['keras_model']
        self._model = Sequential.from_config(config_lstm)
        self._class_name_to_number = params["class_name_to_number"]
        self._target_column_name = params["target_column_name"]
        self._feature_shape = params["feature_shape"]
        self._input_feature_column_name = params["input_feature_column_name"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:

        self._training_inputs = copy.copy(inputs)
        self._training_outputs = copy.copy(outputs)

        if self._training_inputs.shape[0] < self._training_outputs.shape[0]:
            if 'd3mIndex' in self._training_inputs.columns:
                inputs_index_with_contents = self._training_inputs['d3mIndex'].astype(int).tolist()
            else:
                self.logger.warning("No d3mIndex found in input training dataset!")
                inputs_index_with_contents = list(range(self._training_inputs.shape[0]))
            self._training_outputs = self._training_outputs.iloc[inputs_index_with_contents, :]
        elif self._training_inputs.shape[0] > self._training_outputs.shape[0]:
            raise ValueError("The length of inputs is larger than outputs which is impossible.")

        # TODO: maybe use a better way to find the feature input columns
        first_content_index = 0
        input_column_names = []
        input_column_numbers = []
        for i, each_column in enumerate(self._training_inputs.columns):
            while isinstance(self._training_inputs[each_column][first_content_index], type(None)):
                first_content_index += 1
            if type(self._training_inputs[each_column][first_content_index]) is np.ndarray and len(
                    self._training_inputs[each_column][first_content_index].shape) == 2:
                input_column_names.append(each_column)
                input_column_numbers.append(i)
        if len(input_column_names) < 1:
            raise ValueError("No extract feature attribute from input detected!")
        if len(input_column_names) > 1:
            self.logger.warning("More that 1 feature attribute detected! Will only use first one")

        useless_train_rows = []
        input_column_number = input_column_numbers[0]
        for i in range(self._training_inputs.shape[0]):
            if self._training_inputs.iloc[i, input_column_number] is None:
                useless_train_rows.append(i)

        self._training_inputs = self._training_inputs.drop(useless_train_rows)
        self._training_outputs = self._training_outputs.drop(useless_train_rows)

        self.logger.info("following rows are dropped beacuse the training data is None.")
        self.logger.info(str(useless_train_rows))

        self._input_feature_column_name = input_column_names[0]
        self._features = self._training_inputs[self._input_feature_column_name]
        self._training_size = self._training_inputs.shape[0]
        target_column_names = self._training_outputs.columns.tolist()
        if 'd3mIndex' in target_column_names:
            target_column_names.remove('d3mIndex')
        if len(target_column_names) < 1:
            raise ValueError("No target attribute from output detected!")
        if len(target_column_names) > 1:
            self.logger.warning("Multiple target detected! Will only use first one")
        self._target_column_name = target_column_names[0]
        class_names = set(self._training_outputs[self._target_column_name].tolist())
        self._number_of_classes = len(class_names)
        self._feature_shape = list(self._features[first_content_index].shape)
        self._features_amount = self._training_inputs.shape[0]
        self._training_inputs_ndarry = np.empty((self._features_amount, self._feature_shape[0], self._feature_shape[1]))
        for i, each in enumerate(self._features):
            self._training_inputs_ndarry[i] = each
        self._training_ouputs_ndarry = np.zeros((self._features_amount, self._number_of_classes))
        # make a reference from name to number
        for i, each in enumerate(class_names):
            self._class_name_to_number[each] = i
        # update this reference to self._training_ouputs_ndarry
        for i, each in enumerate(self._training_outputs[self._target_column_name]):
            self._training_ouputs_ndarry[i, self._class_name_to_number[each]] = 1
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
            If using the pre-training model, here we will use this model to detect what inside the bounding boxes from training dataset
            Then, count the number of the objects deteced in each box, we will treat only the objects amount number larger than the threshold
            to be the target that we need to detect in the test part.
        """
        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        if timeout is None:
            timeout = np.inf
        if iterations is None:
            iterations = self._epochs
        # if self._minibatch_size > self._training_size:
        #     self._minibatch_size = self._training_size
        # initialze the model if not model loaded
        if not self._model:
            self._model = self._lazy_init_lstm()
        self._iterations_done = 0
        start = time.time()
        self._has_finished = False
        if self._shuffle:
            indices = np.random.permutation(self._training_inputs_ndarry.shape[0])
        else:
            indices = list(range(self._training_inputs_ndarry.shape[0]))
        number_of_training_data = int((1 - self._validate_data_percent) * self._training_size)
        self.logger.info(str(number_of_training_data) + " of the " + str(
            self._training_size) + " input data will be used to trained. The remainede will be used for validation.")
        training_idx, test_idx = indices[:number_of_training_data], indices[number_of_training_data:]
        training_x, test_x = self._training_inputs_ndarry[training_idx, :], self._training_inputs_ndarry[test_idx, :]
        training_y, test_y = self._training_ouputs_ndarry[training_idx, :], self._training_ouputs_ndarry[test_idx, :]

        # repeat fit until interation down or epoch_loss less than threshold
        while time.time() < start + timeout and self._iterations_done < iterations:
            self.logger.info("Start fit on iteration " + str(self._iterations_done))

            result = self._model.fit(training_x, training_y,
                                     # result = self._model.fit(self._training_inputs_ndarry, self._training_ouputs_ndarry,
                                     batch_size=self._batch_size,
                                     # validation_split = self._validate_data_percent,
                                     verbose=self._verbose_mode,
                                     validation_data=(test_x, test_y),
                                     shuffle=self._shuffle,
                                     initial_epoch=self._iterations_done,
                                     epochs=1 + self._iterations_done)
            # result available for these values
            # train_loss = result.history['loss']
            # test_loss   = result.history['val_loss']
            # train_acc  = result.history['acc']
            # test_acc    = result.history['val_acc']
            epoch_loss = result.history['val_loss'][-1]
            self._iterations_done += 1
            if epoch_loss < self._loss_threshold:
                self._has_finished = True
                self.logger.info("The model is well fitted during " + str(self._iterations_done) + "interation. No more needed.")
                return CallResult(None)

        self.logger.info("The model fitting finished with" + str(self._iterations_done) + "interation.")
        self.logger.info("The final result is:")
        try:
            self.logger.info(str(result.history))
            self.logger.info("train_loss      = " + str(result.history['loss']))
            self.logger.info("train_acc       = " + str(result.history['acc']))
            self.logger.info("validation_loss = " + str(result.history['val_loss']))
            self.logger.info("validation_acc  = " + str(result.history['val_acc']))
        except:
            pass
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Use YOLO to detect the objects in the input dataframe
            The function will read the images if the input is a dataframe with image names
            The detected output depends on the input training dataset's ojbects
        """
        if not self._model:
            raise ValueError('The model not fitted or loaded!')

        produce_input_feature = inputs[self._input_feature_column_name]
        result_length = inputs.shape[0]
        produce_input_ndarry = np.empty((result_length, self._feature_shape[0], self._feature_shape[1]))
        for i, each in enumerate(produce_input_feature):
            produce_input_ndarry[i] = each

        prediction = self._model.predict(produce_input_ndarry)
        results = []
        number_to_class_name = dict((v, k) for k, v in self._class_name_to_number.items())
        for each_predict in prediction:
            label = np.argmax(each_predict)
            results.append(number_to_class_name[label])
        output_dataframe = container.DataFrame(inputs['d3mIndex'])
        extracted_feature_dataframe = container.DataFrame({self._target_column_name: results}, generate_metadata=False)
        output_dataframe[self._target_column_name] = extracted_feature_dataframe

        # add metadata
        for each_column in range(2, output_dataframe.shape[1]):
            metadata_selector = (metadata_base.ALL_ELEMENTS, each_column)
            metadata_each_column = {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PredictedTarget',)}
            output_dataframe.metadata = output_dataframe.metadata.update(metadata=metadata_each_column,
                                                                         selector=metadata_selector)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataframe, self._has_finished)

    def _lazy_init_lstm(self):  # -> "tensorflow.keras.models"
        """
            a lazy init function which initialize the LSTM model only when the primitive's fit/ produce method is called
        """
        keras_models = importlib.import_module('tensorflow.keras.models')
        keras_layers = importlib.import_module('tensorflow.keras.layers')
        keras_optimizers = importlib.import_module('tensorflow.keras.optimizers')
        model = keras_models.Sequential()
        # TODO: following parameters could also be hyperparameters for tuning
        model.add(keras_layers.LSTM(self._LSTM_units, return_sequences=False,
                                    input_shape=self._feature_shape,
                                    dropout=self._dropout_rate))
        model.add(keras_layers.Dense(512, activation='relu'))
        model.add(keras_layers.Dropout(self._dropout_rate))
        model.add(keras_layers.Dense(self._number_of_classes, activation='softmax'))
        self._metrics = ['accuracy']
        if self._number_of_classes >= 10:
            self._metrics.append('top_k_categorical_accuracy')
            # Now compile the network
        if self._optimizer_type == 'adam':
            optimizer = keras_optimizers.Adam(lr=self._learning_rate, decay=self._weight_decay)
        elif self._optimizer_type == 'sgd':
            optimizer = keras_optimizers.SGD(lr=self._learning_rate, decay=self._weight_decay, mementum=self._momentum)
        else:
            raise ValueError('Unsupported optimizer_type: {}. Available options: adam, sgd'.format(self._optimizer_type))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=self._metrics)
        return model


'''
    def __getstate__(self) -> typing.Dict:
        """
        This method is used by the pickler as the state of object.
        The object can be recovered through this state uniquely.
        Returns:
            state: Dict
                dictionary of important attributes of the object
        """
        # print("[INFO] Get state called")

        state = self.__dict__  # get attribute dictionary

        # add the fitted_primitives
        state['fitted_pipe'] = self.runtime.steps_state
        state['pipeline'] = self.pipeline.to_json_structure()
        state['log_dir'] = self.log_dir
        state['id'] = self.id
        del state['runtime']  # remove runtime entry

        return state


    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tensorflow.keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = tensorflow.keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    def __setstate__(self, state: typing.Dict) -> None:
        """
        This method is used for unpickling the object. It takes a dictionary
        of saved state of object and restores the object to that state.
        Args:
            state: typing.Dict
                dictionary of the objects picklable state
        Returns:
        """

        # print("[INFO] Set state called!")

        fitted = state['fitted_pipe']
        del state['fitted_pipe']

        structure = state['pipeline']
        state['pipeline'] = Pipeline.from_json_structure(structure)

        run = Runtime(state['pipeline'], fitted_pipeline_id=state['id'],
                      volumes_dir=FittedPipeline.static_volume_dir, log_dir=state['log_dir'])
        run.steps_state = fitted

        state['runtime'] = run

        self.__dict__ = state
'''

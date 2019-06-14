import numpy as np  # type: ignore
import typing
import logging
import importlib


# from pyramid.arima import ARIMA, auto_arima

from . import config


import sklearn
import sklearn.preprocessing
# import tensorflow as tf


from d3m.container.list import List
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata.base import ALL_ELEMENTS

Inputs = d3m_dataframe
Outputs = d3m_dataframe

_logger = logging.getLogger(__name__)


class RNNHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_total_loss(pred_point, pred_lower, pred_upper, label):
        point_loss = tf.reduce_mean(tf.squared_difference(pred_point, label))

        diff_lower = (pred_lower - label)
        diff_p_l = tf.reduce_mean(
            tf.square(tf.clip_by_value(diff_lower, 0, 1e10)))
        diff_n_l = tf.reduce_mean(
            tf.square(tf.clip_by_value(diff_lower, -1e10, 0)))
        lower_loss = diff_p_l * 0.99 + diff_n_l * 0.01

        diff_upper = (pred_upper - label)
        diff_p_u = tf.reduce_mean(
            tf.square(tf.clip_by_value(diff_upper, 0, 1e10)))
        diff_n_u = tf.reduce_mean(
            tf.square(tf.clip_by_value(diff_upper, -1e10, 0)))
        upper_loss = diff_p_u * 0.01 + diff_n_u * 0.99

        total_loss = point_loss + 0.5 * (lower_loss + upper_loss)
        return total_loss


class RNNParams(params.Params):
    target_name: str
    weight: typing.List[np.ndarray]
    n_total: int
    n_valid: int
    n_predict_step: int
    n_train: int


class RNNHyperparams(hyperparams.Hyperparams):
    n_batch = hyperparams.Hyperparameter[int](
        default=1,
        description='Maximum number of batch size',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_max_epoch = hyperparams.Hyperparameter[int](
        default=1000,
        description='Maximum number of Epochs. Default is 300 ',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_max_epoch_total = hyperparams.Hyperparameter[int](
        default=100,
        description='Maximum number of total Epoches. Default is 300 ',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_neurons = hyperparams.Hyperparameter[int](
        default=256,
        description='Neurons in hidden layers',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    n_input_dim = hyperparams.Hyperparameter[int](
        default=1,
        description='Number of input dimension',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_dense_dim = hyperparams.Hyperparameter[int](
        default=128,
        description='Size of fully-connected layers',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_output_dim = hyperparams.Hyperparameter[int](
        default=3,
        description='output dimension',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_patience = hyperparams.Hyperparameter[int](
        default=100,
        description='Number of patience',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    n_lr_decay = hyperparams.Hyperparameter[int](
        default=5,
        description='number of Learning rate decay',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    lr = hyperparams.LogUniform(
        default=1e-2,
        lower=1e-05,
        upper=1,
        description='Learning rate',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    lr_decay = hyperparams.Hyperparameter[float](
        default=0.95,
        description='learning rate decay',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_valid = hyperparams.Hyperparameter[int](
        default=10,
        description='Maximum valid number of iterations. Default is 300 ',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    valid_loss_weight = hyperparams.Hyperparameter[float](
        default=0.5,
        description='Loss weight of validation set',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class RNNTimeSeries(SupervisedLearnerPrimitiveBase[Inputs, Outputs, RNNParams, RNNHyperparams]):
    """
    Timeseries forcasting primitive using recurrent neural network built on tensorflow. Based on code from ISI's SAGE project.
    """
    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "c8d9e1b8-09f0-4b6a-b917-bfbc23f9d90b",
        "version": config.VERSION,
        "name": "DSBox recurrent neural network for timeseries",
        "description": "timeseries forcasting primitive using recurrent neural network built by tensorflow, transferred from ISI's SAGE project",
        "python_path": "d3m.primitives.time_series_forecasting.rnn_time_series.DSBOX",
        "primitive_family": "TIME_SERIES_FORECASTING",
        "algorithm_types": ["RECURRENT_NEURAL_NETWORK"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["feature_extraction",  "timeseries"],
        "installation": [config.INSTALLATION],
        "precondition": ["NO_MISSING_VALUES", "NO_CATEGORICAL_VALUES"],
    })

    def __init__(self, *,
                 hyperparams: RNNHyperparams,
                 random_seed: int = 0,
                 docker_containers: typing.Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)
        self.hyperparams = hyperparams
        self._fitted = False
        self._has_finished = False
        self._iterations_done = False
        self._get_set = False
        self._initialized = False

    def _lazy_init(self):
        global tf
        tf = importlib.import_module("tensorflow")
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
            self.batchX_placeholder = tf.placeholder(
                tf.float32, [self.n_total, self.hyperparams["n_input_dim"]])
            W1 = tf.get_variable('W1', shape=(
                self.hyperparams["n_neurons"], self.hyperparams["n_dense_dim"]),
                initializer=tf.glorot_uniform_initializer())
            b1 = tf.get_variable('b1', shape=(
                1, self.hyperparams["n_dense_dim"]),
                initializer=tf.zeros_initializer())
            W2 = tf.get_variable('W2', shape=(
                self.hyperparams["n_dense_dim"], self.hyperparams["n_output_dim"]),
                initializer=tf.glorot_uniform_initializer())
            b2 = tf.get_variable('b2', shape=(
                1, self.hyperparams["n_output_dim"]),
                initializer=tf.zeros_initializer())

            # Unpack columns
            inputs_series = tf.reshape(self.batchX_placeholder, (1, -1, 1))

            # Forward passes
            self.cell = tf.nn.rnn_cell.GRUCell(self.hyperparams["n_neurons"], kernel_initializer=tf.orthogonal_initializer(
            ), bias_initializer=tf.zeros_initializer())
            self.cell_state = self.cell.zero_state(
                self.hyperparams["n_batch"], dtype=tf.float32)

            states_series, current_state = tf.nn.dynamic_rnn(self.cell, inputs_series, initial_state=self.cell_state,
                                                             parallel_iterations=1)

            prediction = tf.matmul(
                tf.tanh(tf.matmul(tf.squeeze(states_series), W1) + b1), W2) + b2
            self.prediction_method = prediction

            pred_point_train = tf.slice(
                prediction, (0, 0), (self.n_train - self.n_predict_step, 1))
            pred_lower_train = tf.slice(
                prediction, (0, 1), (self.n_train - self.n_predict_step, 1))
            pred_upper_train = tf.slice(
                prediction, (0, 2), (self.n_train - self.n_predict_step, 1))

            pred_point_valid = tf.slice(
                prediction, (self.n_train - self.n_predict_step, 0), (self.n_valid, 1))
            pred_lower_valid = tf.slice(
                prediction, (self.n_train - self.n_predict_step, 1), (self.n_valid, 1))
            pred_upper_valid = tf.slice(
                prediction, (self.n_train - self.n_predict_step, 2), (self.n_valid, 1))

            pred_point_total = tf.slice(
                prediction, (0, 0), (self.n_total - self.n_valid, 1))
            pred_lower_total = tf.slice(
                prediction, (0, 1), (self.n_total - self.n_valid, 1))
            pred_upper_total = tf.slice(
                prediction, (0, 2), (self.n_total - self.n_valid, 1))

            labels_series_train = self.batchX_placeholder[self.n_predict_step: self.n_train, :]
            labels_series_valid = self.batchX_placeholder[self.n_train:, :]
            labels_series_total = self.batchX_placeholder[self.n_predict_step:, :]

            # the total loss take all predictions into account
            self.total_loss_train = RNNHelper.get_total_loss(
                pred_point_train, pred_lower_train, pred_upper_train, labels_series_train)
            self.total_loss_valid = RNNHelper.get_total_loss(
                pred_point_valid, pred_lower_valid, pred_upper_valid, labels_series_valid)
            self.total_loss_total = RNNHelper.get_total_loss(
                pred_point_total, pred_lower_total, pred_upper_total, labels_series_total)
            self.learning_rate = tf.Variable(self.hyperparams["lr"], trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * self.hyperparams["lr_decay"])
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variables = zip(
                *self.optimizer.compute_gradients(self.total_loss_train))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_step = self.optimizer.apply_gradients(
                zip(gradients, variables))
            gradients_total, variables_total = zip(
                *self.optimizer.compute_gradients(self.total_loss_total))
            gradients_total, _ = tf.clip_by_global_norm(gradients_total, 5.0)
            self.train_step_total = self.optimizer.apply_gradients(
                zip(gradients_total, variables_total))
            self.tf_config = tf.ConfigProto()
            self.tf_config.intra_op_parallelism_threads = 1
            self.tf_config.inter_op_parallelism_threads = 1
            # self.session = tf.Session(config=self.tf_config)
            self.saving_path = False
            self._initialized = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)
        if not self._get_set:
            _logger.info("Please set Training Data")
            return CallResult(None)
        self._lazy_init()

        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())

            smallest_loss = float('inf')
            self.smallest_train_loss = float('inf')
            wait = 0
            self._current_cell_state = np.zeros(
                (self.hyperparams["n_batch"], self.hyperparams["n_neurons"]), dtype=np.float32)
            for i in range(self.hyperparams["n_max_epoch"]):
                logging.info(
                    'Epoch: {}/{}'.format(i, self.hyperparams["n_max_epoch"]))
                # train
                train_loss, valid_loss, _train_step = sess.run(
                    [self.total_loss_train, self.total_loss_valid, self.train_step],
                    feed_dict={
                        self.batchX_placeholder: self.x,
                        self.cell_state: self._current_cell_state,
                    }
                )

                sum_loss = train_loss * \
                    (1 - self.hyperparams["valid_loss_weight"]) + \
                    valid_loss * self.hyperparams["valid_loss_weight"]
                logging.info('Epoch {}, Train loss {}, Valid loss {}, Sum loss {}'.format(
                    i, train_loss, valid_loss, sum_loss))
                if wait <= self.hyperparams["n_patience"]:
                    if sum_loss < smallest_loss:
                        smallest_loss = sum_loss
                        self.smallest_train_loss = train_loss
                        self._save_weight(sess)
                        wait = 0
                        logging.info('New smallest')
                    else:
                        wait += 1
                        logging.info('Wait {}'.format(wait))
                        if wait % self.hyperparams["n_lr_decay"] == 0:
                            sess.run(self.learning_rate_decay_op)
                            logging.info('Apply lr decay, new lr: %f' %
                                         self.learning_rate.eval())
                else:
                    break
            self._current_cell_state = np.zeros(
                (self.hyperparams["n_batch"], self.hyperparams["n_neurons"]), dtype=np.float32)
            self._load_weights(sess)
            # if model_saved, loadsaved else  do previously
            _total_loss = sess.run(
                [self.total_loss_total],
                feed_dict={
                    self.batchX_placeholder: self.x,
                    self.cell_state: self._current_cell_state,
                }
            )

            for i in range(self.hyperparams["n_max_epoch_total"]):
                if _total_loss < self.smallest_train_loss:
                    break
                _total_loss, _train_step = sess.run(
                    [self.total_loss_total, self.train_step_total],
                    feed_dict={
                        self.batchX_placeholder: self.x,
                        self.cell_state: self._current_cell_state,
                    }
                )
        self._fitted = True
        return CallResult(None)

    def get_params(self) -> RNNParams:
        if not self._fitted:
            _logger.info("plz fit!")
            return
        return RNNParams(
                            target_name=self._target_name,
                            weight=self.smallest_weight,
                            n_total=self.n_total,
                            n_train=self.n_train,
                            n_valid=self.n_valid,
                            n_predict_step=self.n_predict_step
                        )

        # return RNNParams("./rnn_model.ckpt")

    def set_params(self, *, params: RNNParams) -> None:
        self.smallest_weight = params["weight"]
        self._target_name=params["target_name"]
        self.n_total=params["n_total"]
        self.n_valid=params["n_valid"]
        self.n_predict_step=params["n_predict_step"]
        self.n_train=params["n_train"]
        self._lazy_init()

        # open this file for loading

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        ########################################################
        #  need to use inputs to figure out the params of RNN  #
        ########################################################
        # self._lazy_init()
        if len(inputs) == 0:
            _logger.info(
                "Warning: Inputs timeseries data to timeseries_featurization primitive's length is 0.")
            return

        self.n_predict_step = self.hyperparams["max_valid"]
        self.scaler = sklearn.preprocessing.StandardScaler()
        if 'd3mIndex' in outputs.columns:
            self._index = outputs['d3mIndex']
            self._training_inputs = outputs.drop(columns=['d3mIndex'])# Arima takes shape(n,) as inputs, only target casting is applied
        else:
            self._index = inputs['d3mIndex']
            self._training_inputs = outputs
        data = self._training_inputs.values
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        self._target_name = outputs.columns[-1]

        self.x = data_scaled.reshape(-1, 1)
        self.n_valid = self.n_predict_step
        self.n_train = len(self.x) - self.n_valid
        self.n_total = len(self.x)
        self._get_set = True

    def _save_weight(self, sess):
        tf_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)  # tf_vars
        self.smallest_weight = sess.run(tf_vars)

    def _load_weights(self, sess):
        tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ops = []
        for i_tf in range(len(tf_vars)):
            ops.append(tf.assign(tf_vars[i_tf], self.smallest_weight[i_tf]))
        sess.run(ops)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            _logger.info("Please run fit() first!")
            return CallResult(None, self._has_finished, self._iterations_done)

        self.n_predict_step = inputs.shape[0]
        self.pred_point_test = tf.slice(
            self.prediction_method, (self.n_total - self.n_predict_step, 0), (self.n_predict_step, 1))
        self.pred_lower_test = tf.slice(
            self.prediction_method, (self.n_total - self.n_predict_step, 1), (self.n_predict_step, 1))
        self.pred_upper_test = tf.slice(
            self.prediction_method, (self.n_total - self.n_predict_step, 2), (self.n_predict_step, 1))

        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            self._load_weights(sess)
            self._current_cell_state = np.zeros(
                (self.hyperparams["n_batch"], self.hyperparams["n_neurons"]), dtype=np.float32)
            pred_test, pred_test_lower, pred_test_upper = sess.run(
                [self.pred_point_test, self.pred_lower_test, self.pred_upper_test],
                feed_dict={
                    self.batchX_placeholder: self.x,
                    self.cell_state: self._current_cell_state,
                }
            )
            pred_test = self.scaler.inverse_transform(pred_test)
            pred_test_lower = self.scaler.inverse_transform(pred_test_lower)
            pred_test_upper = self.scaler.inverse_transform(pred_test_upper)
            pred_test_lower = np.minimum(
                pred_test, np.minimum(pred_test_lower, pred_test_upper))
            pred_test_upper = np.maximum(
                pred_test, np.maximum(pred_test_upper, pred_test_lower))

            _logger.info(pred_test.tolist())

        res=pred_test.ravel().tolist()
        output = d3m_dataframe({'d3mIndex': inputs['d3mIndex'], self._target_name: res})
        output.metadata = inputs.metadata.clear(
            source=self, for_value=output, generate_metadata=True)
        meta_d3mIndex = {"name": "d3mIndex", "structural_type":int, "semantic_types":["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/PrimaryKey"]}
        meta_target = {"name": self._target_name, "structural_type":float, "semantic_types":["https://metadata.datadrivendiscovery.org/types/Target", "https://metadata.datadrivendiscovery.org/types/PredictedTarget"]}
        output.metadata = output.metadata.update(selector=(ALL_ELEMENTS, 0), metadata=meta_d3mIndex)
        output.metadata = output.metadata.update(selector=(ALL_ELEMENTS, 1), metadata=meta_target)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output, self._has_finished, self._iterations_done)

    @classmethod
    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                                   source: typing.Any,  target_names: typing.List = None,) -> metadata_base.DataMetadata:
        for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  source=source)
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  source=source)
            if target_names:
                metadata = metadata.update((metadata_base.ALL_ELEMENTS, column_index), {
                    'name': target_names[column_index],
                }, source=source)
        return metadata

if __name__ == "__main__":
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9,
          1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9]
    h = 5
    R = RNNTimeSeries()
    R.produce(inputs=List(ts))

import numpy as np  # type: ignore
import logging
import typing

from pmdarima.arima import ARIMA, auto_arima
from collections import defaultdict
from . import config

from d3m.metadata.base import DataMetadata, ALL_ELEMENTS
from d3m.container import DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata.base import ALL_ELEMENTS
import common_primitives.utils as common_utils

ONE_TIME_SERIES_KEY = 'only_one_time_series'
# Inputs = container.List
Inputs = DataFrame
Outputs = DataFrame
_logger = logging.getLogger(__name__)


class ArimaParams(params.Params):
    arima: ARIMA
    target_name: str


class ArimaHyperparams(hyperparams.Hyperparams):
    auto = hyperparams.Hyperparameter[bool](
        default=True,
        description="Use Auto fit or not for ARIMA",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    P = hyperparams.Hyperparameter[int](
        default=0,
        description="the order (number of time lags) of the auto-regressive model",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    D = hyperparams.Hyperparameter[int](
        default=0,
        description="the degree of differencing (the number of times the data have had past values subtracted)",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    Q = hyperparams.Hyperparameter[int](
        default=1,
        description="the order of the moving-average model",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    is_seasonal = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether it is seasonal"
    )
    seasonal_order = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(0, 1, 2, 12),
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="seasonal component of AR parameters"
    )
    # start_params = hyperparams.Set(
    #     elements=hyperparams.Hyperparameter[int](-1),
    #     default=(),
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters to use. ",

    # )
    transparams = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether or not to transform parameters to ensure stationarity",
    )
    method = hyperparams.Enumeration(
        values=[None, "css-mle", "mle", "css"],
        default=None,
        description='the loglikelihood to maximize',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    trend = hyperparams.Enumeration[str](
        values=['c', 'ct', 't'],
        default='c',
        description='trend polynomial A(t)',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    solver = hyperparams.Enumeration[str](
        values=['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'],
        default='lbfgs',
        description='solver to be used',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    maxiter = hyperparams.Hyperparameter[int](
        default=50,
        description="max_iter",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    disp = hyperparams.Hyperparameter[int](
        default=0,
        description="disp controls the frequency of the output during the iterations",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    callback = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="trace type",

    )
    suppress_warnings = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="suppress_warnings",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='append',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


class AutoArima(SupervisedLearnerPrimitiveBase[Inputs, Outputs, ArimaParams, ArimaHyperparams]):
    """
    ARIMA primitive for timeseries data regression/forecasting problems. Based on Pyarmid/Arima.
    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        # Required
        "id": 'b2e4e8ea-76dc-439e-8e46-b377bf616a35',
        "version": config.VERSION,
        "name": "DSBox Arima Primitive",
        "description": "Arima primitive for timeseries data regression/forcasting problems, transferred from pyramid/Arima",

        "python_path": "d3m.primitives.time_series_forecasting.arima.DSBOX",
        "primitive_family": "TIME_SERIES_FORECASTING",
        "algorithm_types": ["AUTOREGRESSIVE_INTEGRATED_MOVING_AVERAGE"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["Transform", "Timeseries", "Aggregate"],
        "installation": [config.INSTALLATION],
        "precondition": ["NO_MISSING_VALUES", "NO_CATEGORICAL_VALUES"],
    })

    def __init__(self, *,
                 hyperparams: ArimaHyperparams,
                 random_seed: int = 0,
                 docker_containers: typing.Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)
        self._index = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_name = None
        self._fitted = False
        self._time_series = dict()
        self._length_for_produce = 0
        self._key_columns = []
        self._index_and_value_columns = []
        self._last_time = dict()
        self._model = dict()

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        ########################################################
        # need to use inputs to figure out the params of ARIMA #
        ########################################################
        self._training_inputs = copy.deepcopy(inputs)
        self._training_outputs = copy.deepcopy(outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)
        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        if len(self._training_inputs) == 0:
            _logger.error("Inputs timeseries data to timeseries_featurization primitive's length is 0.")
            return

        for i in range(self._training_inputs.shape[1]):
            each_meta = inputs.metadata.query((ALL_ELEMENTS, i))
            # TODO: in stock market, "Year" column is also a CategoricalData???
            if "https://metadata.datadrivendiscovery.org/types/CategoricalData" in each_meta['semantic_types'] and "https://metadata.datadrivendiscovery.org/types/Time" not in each_meta['semantic_types']:
                self._key_columns.append(i)

            if "https://metadata.datadrivendiscovery.org/types/TrueTarget" in each_meta['semantic_types'] or "https://metadata.datadrivendiscovery.org/types/Time" in each_meta['semantic_types']:
                self._index_and_value_columns.append(i)

        if len(self._index_and_value_columns) == 0:
            _logger.info("No CategoricalData column found, will treat as only 1 timeseries input")
            self._fit_single_timeseries()
        else:
            _logger.info("CategoricalData column found, will treat as multiple timeseries input")
            self._fit_multiple_timeseries()

        target_columns = res3.value.metadata.list_columns_with_semantic_types(("https://metadata.datadrivendiscovery.org/types/SuggestedTarget",))
        if len(target_columns) > 1:
            _logger.warning("More than 1 target columns found! Will ignore the others")
        elif len(target_columns) == 0:
            _logger.error("No target found in training output data!")
            return CallResult(None)
        self._target_name = self._training_outputs.columns[target_columns[0]]

        if self.hyperparams["auto"]:
            for key, val in self._time_series.items():
                self._model[key] = auto_arima(val)
            self._fitted = True
            return CallResult(None)

        else:
            if self.hyperparams["is_seasonal"]:
                seasonal_order = self.hyperparams["seasonal_order"]
            else:
                seasonal_order = None
            for key, val in self._time_series.items():
                each_model = ARIMA(
                        order=(
                            self.hyperparams["P"],
                            self.hyperparams["D"], self.hyperparams["Q"]
                        ),
                        seasonal_order=seasonal_order,
                        transparams=self.hyperparams["transparams"],
                        method=self.hyperparams["method"],
                        trend=self.hyperparams["trend"],
                        solver=self.hyperparams["solver"],
                        maxiter=self.hyperparams["maxiter"],
                        disp=self.hyperparams["disp"],
                        # callback=self.hyperparams["callback"],
                        callback=None,
                        suppress_warnings=self.hyperparams["suppress_warnings"],
                        out_of_sample_size=False,
                        scoring="mse",
                        scoring_args=None
                )
                each_model.fit(val)
                self._model[key] = each_model
            self._fitted = True

            return CallResult(None)

    def _fit_single_timeseries(self) -> None:
        """
        Inner function used to generate the timesequence with only 1 timeseries condition (e.g. 56_sunspot)
        It will add the formated timesequence to self._time_series for further use
        """
        # Arima takes shape(n,) as inputs, only target casting is applied
        # if 'd3mIndex' in self._training_outputs.columns:
        #     self._index = self._training_outputs['d3mIndex']
        #     temp = self._training_outputs.drop(columns=['d3mIndex'])
        # else:
        #     temp = self._training_outputs
            
        temp = self._process_time_unit(self._training_outputs)
        temp = temp.sort_values(by=['time_unit'])
        previous_time = None
        previous_val = None
        sequences = []
        for _, each_row in temp.iterrows():
            current_time = int(each_row["time_unit"])
            current_val = val_type(each_row["count"])
            if not previous_time:
                previous_time = current_time
                previous_val = current_val
                sequences.append(current_val)
                continue
            diff = current_time - previous_time
            if diff > 1:
                for i in range(1, diff):
                    new_val = val_type((current_val - previous_val) / diff * i + previous_val)
                    sequences.append(new_val)
            sequences.append(current_val)
            previous_time = current_time
            previous_val = current_val
        self._last_time[ONE_TIME_SERIES_KEY] = current_time
        self._time_series[ONE_TIME_SERIES_KEY] = sequences

    def _fit_multiple_timeseries(self) -> None:
        """
        Inner function used to generate the timesequence with more than timeseries condition (e.g. LL1_736_population_spawn)
        It will add the formated timesequence to self._time_series for further use
        """
        self._training_inputs = self._process_time_unit(self._training_inputs)
        records = defaultdict(list)
        for row_number, each_row in self._training_inputs.iterrows():
            key_name = []
            for each in self._key_columns:
                key_name.append(each_row[each])
            records[tuple(key_name)].append(row_number)

        val_type = self._training_outputs[self._target_name].dtype.name
        if "int" in val_type:
            val_type = int
        else:
            val_type = float

        for key, val in records.items():
            temp = self._training_inputs.iloc[val, :]
            temp = temp.sort_values(by=['time_unit'])
            previous_time = None
            previous_val = None
            sequences = []
            for _, each_row in temp.iterrows():
                current_time = int(each_row["time_unit"])
                current_val = val_type(each_row["count"])
                if not previous_time:
                    previous_time = current_time
                    previous_val = current_val
                    sequences.append(current_val)
                    continue
                diff = current_time - previous_time
                if diff > 1:
                    for i in range(1, diff):
                        new_val = val_type((current_val - previous_val) / diff * i + previous_val)
                        sequences.append(new_val)
                sequences.append(current_val)
                previous_time = current_time
                previous_val = current_val
            # record to last time appeared in training data
            self._last_time[key] = current_time
            self._time_series[key] = sequences

    def _process_time_unit(self, input_df) -> DataFrame:
        """
        Inner function used to process a input df's Time data. 
        It will transform the input Time format data into a 3 columns output.
        With a format as:
        First column as d3mIndex
        Second column as Time unit called "time_unit" in int format
        Thid column as value unit called "count" in int or float format
        """


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        if len(self._model) == 1:
            output = self._produce_single_timeseries(inputs)
        else:
            output = self._produce_multiple_timeseries(inputs)

        return output

    def _produce_single_timeseries(self, inputs) -> "CallResult":
        # if self.hyperparams['use_semantic_types'] is True:
        #     sk_inputs = inputs.iloc[:, self._training_indices]

        inputs_copy = copy.deepcopy(inputs)
        inputs_processed = self._process_time_unit(inputs_copy)
        inputs_processed = inputs_processed.sort_values(by=['time_unit'])
        test_last_time = inputs_processed.iloc[-1]['time_unit']
        train_last_time = self._last_time[ONE_TIME_SERIES_KEY]
        n_periods = test_last_time - train_last_time
        res = self._model[ONE_TIME_SERIES_KEY].predict(n_periods=n_periods)

        for i in range(inputs_processed.shape[0]):
            val = res[inputs_processed.iloc[i]['time_unit'] - train_last_time]
            inputs_processed.iloc[i]['count'] = val

        inputs_processed = inputs_processed.set_index('d3mIndex')

        # transform inputs_processed back to the original intput base on d3mIndex
        output = copy.deepcopy(inputs)
        for i in range(output.shape[0]):
            index = output.iloc[i]['d3mIndex']
            output.iloc[i][self._target_name] = inputs_processed.iloc[index]['count']

        # maybe still need these part of codes

        # if isinstance(output, DataFrame):
        #     output.metadata = output.metadata.clear(source=self, for_value=output, generate_metadata=True)
        #     meta_d3mIndex = {"name": "d3mIndex", "structural_type":int, "semantic_types":["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/PrimaryKey"]}
        #     meta_target = {"name": self._target_name, "structural_type":float, "semantic_types":["https://metadata.datadrivendiscovery.org/types/Target", "https://metadata.datadrivendiscovery.org/types/PredictedTarget"]}
        #     output.metadata = output.metadata.update(selector=(ALL_ELEMENTS, 0), metadata=meta_d3mIndex)
        #     output.metadata = output.metadata.update(selector=(ALL_ELEMENTS, 1), metadata=meta_target)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output, self._has_finished, self._iterations_done)

    def _produce_multiple_timeseries(self, inputs) -> "CallResult":
        """
        same as produce_single_timeseries, just for condition of multiple timeseries
        """


    def get_params(self) -> ArimaParams:
        return ArimaParams(arima=self._model, target_name=self._target_name)

    def set_params(self, *, params: ArimaParams) -> None:
        self._model = params["arima"]
        self._target_name = params["target_name"]

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: ArimaHyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                                     use_columns=hyperparams['use_columns'],
                                                                                     exclude_columns=hyperparams[
                                                                                         'exclude_columns'],
                                                                                     can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: ArimaHyperparams) -> bool:
        column_metadata = inputs_metadata.query(
            (metadata_base.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    @classmethod
    def _get_targets(cls, data: DataFrame, hyperparams: ArimaHyperparams):
        if not hyperparams['use_semantic_types']:
            return data, []
        target_names = []
        target_column_indices = []
        metadata = data.metadata
        target_column_indices.extend(metadata.get_columns_with_semantic_type(
            'https://metadata.datadrivendiscovery.org/types/TrueTarget'))

        for column_index in target_column_indices:
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_index = typing.cast(
                metadata_base.SimpleSelectorSegment, column_index)
            column_metadata = metadata.query(
                (metadata_base.ALL_ELEMENTS, column_index))
            target_names.append(column_metadata.get('name', str(column_index)))

        targets = data.iloc[:, target_column_indices]
        return targets, target_names

    @classmethod
    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                                   source: typing.Any,  target_names: typing.List, target_column_number: typing.List) -> metadata_base.DataMetadata:
        for column_index in target_column_number:
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
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 5, 6,
          7, 8, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2,
          3, 4, 5, 6, 7, 8, 8, 9]
    h = 5
    model = auto_arima(ts)
    res = model.predict(n_periods=h)

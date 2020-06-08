import copy
import logging
import typing

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from collections import OrderedDict

from pmdarima.arima import ARIMA, auto_arima  # type: ignore

from d3m.base import utils as base_utils
from d3m.container import DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import d3m.metadata.base as mbase
import common_primitives.utils as common_utils

from . import config
from .utils import TimeIndicator, ExtractTimeseries


ONE_TIME_SERIES_KEY = 'only_one_time_series'
PRIMARY_KEY = "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
TRUE_TARGET_TYPE = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
# SUGGESTED_TARGET_TYPE = 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'

# Inputs = container.List
Inputs = DataFrame
Outputs = DataFrame


class ArimaParams(params.Params):
    last_time: typing.Dict
    model: typing.Dict
    attribute_columns_names: typing.List[str]
    target_columns_metadata: typing.List[OrderedDict]
    target_columns_names: typing.List[str]
    time_indicator: TimeIndicator
    input_data_processed: typing.Dict


class ArimaHyperparams(hyperparams.Hyperparams):
    take_log = hyperparams.Hyperparameter[bool](
        default=True,
        description="Whether take log to inputs timeseries",
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ]
    )

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
        self._fitted = False
        self._length_for_produce = 0
        self._last_time: typing.Dict = dict()
        self._model = dict()
        self._attribute_columns_names: typing.List[str] = None
        self._target_columns_metadata: typing.List[OrderedDict] = None
        self._target_columns_names: typing.List[str] = None
        self._time_indicator: TimeIndicator = None
        self._extract_timeseries: ExtractTimeseries = None
        self._input_data_processed = dict()
        self.logger = logging.getLogger(__name__)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        ########################################################
        # need to use inputs to figure out the params of ARIMA #
        ########################################################
        # if len(inputs) == 0:
        #     self.logger.info(
        #         "Warning: Inputs timeseries data to timeseries_featurization primitive's length is 0.")
        #     return
        # if 'd3mIndex' in outputs.columns:
        #     self._index = outputs['d3mIndex']
        #     self._training_inputs = outputs.drop(columns=['d3mIndex'])# Arima takes shape(n,) as inputs, only target casting is applied
        # else:
        #     self._training_inputs = outputs
        self._store_columns_metadata_and_names(inputs, outputs)
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._time_indicator = TimeIndicator(self._training_inputs)
        self._extract_timeseries = ExtractTimeseries(self._training_inputs, self._time_indicator)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)
        if self._training_inputs is None:
            raise ValueError("Missing training data.")
        
        target_column_names = []
        for i in range(self._training_inputs.shape[1]):
            each_selector = (metadata_base.ALL_ELEMENTS, i)
            each_column_meta = self._training_inputs.metadata.query(each_selector)
            if "https://metadata.datadrivendiscovery.org/types/TrueTarget" in each_column_meta["semantic_types"]:
                target_column_names.append(self._training_inputs.columns[i])

        if len(target_column_names) > 1:
            self.logger.warning("Multiple target detected!")
        target_column_name = target_column_names[0]

        for name, one_timeseries_df in self._extract_timeseries.groupby():
            print('fitting', name, 'with', one_timeseries_df.shape)
            self._input_data_processed[name] = one_timeseries_df[target_column_name]
            self._last_time[name] = one_timeseries_df.index[-1]

            # For now, just use the target
            one_timeseries = one_timeseries_df.iloc[:, -1].values
            if self.hyperparams["take_log"]:
                one_timeseries = np.log(one_timeseries)
            if self.hyperparams["auto"]:
                self._model[name] = auto_arima(one_timeseries, error_action="ignore")
            else:
                if self.hyperparams["is_seasonal"]:
                    seasonal_order = self.hyperparams["seasonal_order"]
                else:
                    seasonal_order = None
                self._model[name] = ARIMA(
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
                self._model[name].fit(one_timeseries)
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        inputs0 = copy.deepcopy(inputs)
        test_extract_timeseries = ExtractTimeseries(inputs0, self._time_indicator)

        results: typing.Dict = {}
        for name, one_timeseries_df in test_extract_timeseries.groupby():
            last_training_time = self._last_time[name]
            last_produce_time = one_timeseries_df.index[-1]
            n_periods = int(self._time_indicator.get_difference(last_training_time, last_produce_time))
            if n_periods <= 0:
                # self.logger.error("processing on " + str(name) + "failed!")
                # self.logger.error("last training time is" + str(last_training_time))
                # self.logger.error("last produce time is, " + str(last_produce_time))
                # self.logger.error(str(one_timeseries_df))
                # self.logger.error(str(inputs0))
                # _
                self.logger.warning(
                    'Testing period is not after training period! last_training_time={last_training_time} last_produce_time={last_produce_time}'.format(
                    last_training_time=str(last_training_time), 
                    last_produce_time=str(last_produce_time))
                )
                prediction = []
                # return CallResult(inputs)
                # n_periods = 1
            else:
                prediction = self._model[name].predict(n_periods=n_periods)
            if self.hyperparams['take_log']:
                prediction = np.exp(prediction)

            # Need to pick out rows based on time
            start = self._time_indicator.get_next_time(last_training_time)
            # in our ta2 system, we will meet this condition
            if start > last_produce_time:
                last_produce_time = start
            all_periods = pd.DataFrame(prediction, index=self._time_indicator.get_date_range(start, last_produce_time))
            result = []
            dates = []
            for date in one_timeseries_df.index:
                if isinstance(date, pd.Timestamp):
                    dates.append(date.to_pydatetime())
                else:
                    dates.append(date)
                if date in all_periods.index:
                    # find result from prediction of arima
                    result.append(all_periods.loc[date, 0])
                else:
                    # find results from training data
                    if date in self._input_data_processed[name].index:
                        temp = self._input_data_processed[name].loc[date]
                    else:
                        for i, each_index in enumerate(self._input_data_processed[name].index):
                            if date > each_index:
                                if i == 0:
                                    temp = self._input_data_processed[name].loc[i]
                                else:
                                    temp_left = self._input_data_processed[name].loc[i-1]
                                    temp_right = self._input_data_processed[name].loc[i]
                                    temp = (temp_left + temp_right) / 2

                    result.append(temp)
            result_df = pd.DataFrame(result, index=dates)
            if isinstance(name, str):
                key_tuple = tuple([name])
            else:
                key_tuple = tuple(name)
            results[key_tuple] = result_df

        prediction_vector = test_extract_timeseries.combine(results, inputs0)
        output_columns = [self._wrap_predictions(prediction_vector)]
        outputs = base_utils.combine_columns(inputs, [], output_columns, return_result='new', add_index_columns=True)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(outputs)

    def get_params(self) -> ArimaParams:
        return ArimaParams(
            last_time=self._last_time,
            model=self._model,
            attribute_columns_names=self._attribute_columns_names,
            target_columns_metadata=self._target_columns_metadata,
            target_columns_names=self._target_columns_names,
            time_indicator=self._time_indicator,
            input_data_processed = self._input_data_processed,
        )

    def set_params(self, *, params: ArimaParams) -> None:
        # self._model = params["arima"]
        # self._target_name = params["target_name"]
        self._last_time = params['last_time']
        self._model = params['model']
        self._attribute_columns_names = params['attribute_columns_names']
        self._target_columns_metadata = params['target_columns_metadata']
        self._target_columns_names = params['target_columns_names']
        self._time_indicator = params['time_indicator']
        self._input_data_processed = params['input_data_processed']


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
        target_names: typing.List = []
        target_column_indices: typing.List = []
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

    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata) -> typing.List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: typing.List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = list(column_metadata.get('semantic_types', []))
            if 'https://metadata.datadrivendiscovery.org/types/PredictedTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            semantic_types = [semantic_type for semantic_type in semantic_types if semantic_type != 'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            column_metadata['semantic_types'] = semantic_types

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    def _store_columns_metadata_and_names(self, inputs: Inputs, outputs: Outputs) -> None:
        self._attribute_columns_names = list(inputs.columns)
        self._target_columns_metadata = self._get_target_columns_metadata(outputs.metadata)
        self._target_columns_names = list(outputs.columns)

    @classmethod
    def _update_predictions_metadata(cls, outputs: typing.Optional[Outputs], target_columns_metadata: typing.List[OrderedDict]) -> metadata_base.DataMetadata:
        outputs_metadata = metadata_base.DataMetadata()
        if outputs is not None:
            outputs_metadata = outputs_metadata.generate(outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, predictions: np.ndarray) -> Outputs:
        outputs = DataFrame(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(outputs, self._target_columns_metadata)
        outputs.columns = self._target_columns_names
        return outputs


if __name__ == "__main__":
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 5, 6,
          7, 8, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2,
          3, 4, 5, 6, 7, 8, 8, 9]
    h = 5
    model = auto_arima(ts)
    res = model.predict(n_periods=h)

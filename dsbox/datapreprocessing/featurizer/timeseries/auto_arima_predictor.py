import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import stopit
import math
import typing
from typing import Any, Callable, List, Dict, Union, Optional
import logging
import copy

from pyramid.arima import ARIMA, auto_arima

from . import config

from d3m.container.list import List
from d3m.container.numpy import ndarray
from d3m.container import DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.primitive_interfaces.base import CallResult, DockerContainer, MultiCallResult
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin


# Inputs = container.List
Inputs = DataFrame
Outputs = DataFrame
_logger = logging.getLogger(__name__)

class ArimaParams(params.Params):
    arima: ARIMA


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
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)
        self._index = None
        self._training_inputs = None
        self._target_name = None
        self._fitted = False
        self._length_for_produce = 0

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        ########################################################
        # need to use inputs to figure out the params of ARIMA #
        ########################################################
        if len(inputs) == 0:
            _logging.info(
                "Warning: Inputs timeseries data to timeseries_featurization primitive's length is 0.")
            return
        if 'd3mIndex' in outputs.columns:
            self._index = outputs['d3mIndex']
            self._training_inputs = outputs.drop(columns=['d3mIndex'])# Arima takes shape(n,) as inputs, only target casting is applied
        else:
            self._training_inputs = outputs
        self._target_name = self._training_inputs.columns[-1]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)
        if self._training_inputs is None:
            raise ValueError("Missing training data.")
        if self.hyperparams["take_log"]:
            self._training_inputs = np.log(self._training_inputs.value.ravel())
        if self.hyperparams["auto"]:
            self._model = auto_arima(self._training_inputs)
            self._fitted = True
            return CallResult(None)
        else:
            if self.hyperparams["is_seasonal"]:
                seasonal_order = self.hyperparams["seasonal_order"]
            else:
                seasonal_order = None
            self._model = ARIMA(
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
            self._model.fit(self._training_output)
            self._fitted = True

            return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        n_periods = inputs.shape[0]
        if 'd3mIndex' in inputs:
            self._index = inputs['d3mIndex']
        if self.hyperparams['take_log']:
            inputs = np.log(inputs.value.ravel())
        if self.hyperparams['use_semantic_types'] is True:
            sk_inputs = inputs.iloc[:, self._training_indices]
        res_df = self._model.predict(n_periods=n_periods)
        if self.hyperparams['take_log']:
            res_df = np.exp(res_df)
        if self._index is not None:
            output = DataFrame({'d3mIndex': self._index, self._target_name: res_df})
        if isinstance(output, DataFrame):
            output.metadata = output.metadata.clear(
                source=self, for_value=output, generate_metadata=True)
            output.metadata = self._add_target_semantic_types(
                metadata=output.metadata, target_names=self._target_name, source=self)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output, self._has_finished, self._iterations_done)

    def get_params(self) -> ArimaParams:
        return ArimaParams(arima=self._model)

    def set_params(self, *, params: ArimaParams) -> None:
        self._model = params["arima"]

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
                                   source: typing.Any,  target_names: List = None,) -> metadata_base.DataMetadata:
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
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2, 3, 4, 5, 6,
          7, 8, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 1, 2,
          3, 4, 5, 6, 7, 8, 8, 9]
    h = 5
    model = auto_arima(ts)
    res = model.predict(n_period=h)

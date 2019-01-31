import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import stopit
import math
import typing
from typing import Any, Callable, List, Dict, Union, Optional


from pyramid.arima import ARIMA, auto_arima

from . import config

from d3m.container.list import List
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.primitive_interfaces.base import CallResult, DockerContainer, MultiCallResult
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin


# Inputs = container.List
Inputs = d3m_dataframe
Outputs = d3m_dataframe


class ArimaParams(params.Params):
    arima: ARIMA


class ArimaHyperparams(hyperparams.Hyperparams):
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
        default=True,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether it is seasonal",
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
    __author__ = 'Pada'
    metadata = hyperparams.base.PrimitiveMetadata({
        # Required
        "id": 'b2e4e8ea-76dc-439e-8e46-b377bf616a35',
        "version": config.VERSION,
        "name": "DSBox Arima Primitive",
        "description": "Arima for timeseries data regression/forcasting problem",

        "python_path": "d3m.primitives.time_series_forecasting.Arima.DSBOX",
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
        if self.hyperparams["is_seasonal"]:
            seasonal_order = self.hyperparams["seasonal_order"]
        else:
            seasonal_order = None
        self._clf = ARIMA(
            order=(self.hyperparams["P"],
                   self.hyperparams["D"], self.hyperparams["Q"]),
            seasonal_order=seasonal_order,
            # seasonal_order=self.hyperparams["seasonal_order"],
            # seasonal_order=(0,1,1,12),
            # start_params=self.hyperparams["start_params"],
            # start_params = None,
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
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._fitted = False
        self._length_for_produce = 0

    def set_training_data(self, *, inputs: Inputs) -> None:
        inputs_timeseries = d3m_dataframe(inputs.iloc[:, -1])
        inputs_d3mIndex = d3m_dataframe(inputs.iloc[:, 0])
        if len(inputs_timeseries) == 0:
            print(
                "Warning: Inputs timeseries data to timeseries_featurization primitive's length is 0.")
            return
        column_name = inputs_timeseries.columns[0]
        self._training_inputs, self._target_names = inputs_timeseries, column_name
        self._training_outputs = inputs_timeseries

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")
        arima_training_output = d3m_ndarray(self._training_outputs)

        shape = arima_training_output.shape
        if len(shape) == 2 and shape[1] == 1:
            sk_training_output = np.ravel(arima_training_output)

        self._clf.fit(sk_training_output)
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        arima_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        sk_output = self._clf.predict(n_periods=len(arima_inputs))
        output = d3m_dataframe(sk_output, generate_metadata=False, source=self)
        output.metadata = inputs.metadata.clear(
            source=self, for_value=output, generate_metadata=True)
        output.metadata = self._add_target_semantic_types(
            metadata=output.metadata, target_names=self._target_names, source=self)
        if not self.hyperparams['use_semantic_types']:
            return CallResult(output)
        # outputs = common_utils.combine_columns(return_result=self.hyperparams['return_result'],
        #                                        add_index_columns=self.hyperparams['add_index_columns'],
        #                                        inputs=inputs, column_indices=self._training_indices, columns_list=[output], source=self)

        return CallResult(output)

    def get_params(self) -> ArimaParams:
        return Params(arima=self._clf)

    def set_params(self, *, params: ArimaParams) -> None:
        self._clf = params["arima"]

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
    def _get_targets(cls, data: d3m_dataframe, hyperparams: ArimaHyperparams):
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

# functions to fit in devel branch of d3m (2019-1-17)
    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        This method allows primitive author to implement an optimized version of both fitting
        and producing a primitive on same data.

        If any additional method arguments are added to primitive's ``set_training_data`` method
        or produce method(s), or removed from them, they have to be added to or removed from this
        method as well. This method should accept an union of all arguments accepted by primitive's
        ``set_training_data`` method and produce method(s) and then use them accordingly when
        computing results.

        The default implementation of this method just calls first ``set_training_data`` method,
        ``fit`` method, and all produce methods listed in ``produce_methods`` in order and is
        potentially inefficient.

        Parameters
        ----------
        produce_methods : Sequence[str]
            A list of names of produce methods to call.
        inputs : Inputs
            The inputs given to ``set_training_data`` and all produce methods.
        outputs : Outputs
            The outputs given to ``set_training_data``.
        timeout : float
            A maximum time this primitive should take to both fit the primitive and produce outputs
            for all produce methods listed in ``produce_methods`` argument, in seconds.
        iterations : int
            How many of internal iterations should the primitive do for both fitting and producing
            outputs of all produce methods.

        Returns
        -------
        MultiCallResult
            A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)


if __name__ == "__main__":
    pass

'''
Created on Jan 23, 2018

@author: kyao
'''

import numpy as np
import typing
import logging
import frozendict
from d3m.metadata import hyperparams, params
from d3m import container
from d3m.exceptions import InvalidArgumentValueError
import d3m.metadata.base as mbase
from d3m.container.pandas import DataFrame as d3m_DataFrame
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection

# from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase
# changed primitive class to fit in devel branch of d3m (2019-1-17)
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import pandas as pd
from . import config

Inputs = container.List#[container.DataFrame]  # this format is for old version of d3m
Outputs = container.DataFrame
_logger = logging.getLogger(__name__)

class Params(params.Params):
    x_dim: int
    y_dim: int
    value_dimension: int
    projection_param: typing.Dict
    components_: typing.Optional[np.ndarray]
    value_found: bool


class Hyperparams(hyperparams.Hyperparams):
    '''
    eps : Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
    '''
    eps = hyperparams.Uniform(
        lower=0.1,
        upper=0.5,
        default=0.2,
        semantic_types=["http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
        )

    generate_metadata = hyperparams.UniformBool(
        default = True,
        description="A control parameter to set whether to generate metada after the feature extraction. It will be very slow if the columns length is very large. For the default condition, it will turn off to accelerate the program running.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )


class RandomProjectionTimeSeriesFeaturization(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    Timeseries collection featurization using random projection.
    '''

    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox.timeseries_featurization.random_projection",
        "version": config.VERSION,
        "name": "DSBox random projection timeseries featurization ",
        "description": "A simple timeseries featurization using random projection",
        "python_path": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
        "primitive_family": "FEATURE_EXTRACTION",
        "algorithm_types": [ "RANDOM_PROJECTION" ],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [ config.REPOSITORY ]
            },
        ### Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        ### Optional
        "keywords": [ "feature_extraction",  "timeseries"],
        "installation": [ config.INSTALLATION ],
        #"location_uris": [],
        "precondition": ["NO_MISSING_VALUES", "NO_CATEGORICAL_VALUES"],
        "effects": ["NO_JAGGED_VALUES"],
        #"hyperparms_to_tune": []
        })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

        self._model = None
        self._training_data = None
        self._value_found = False
        self._x_dim = 0  # x_dim : the amount of timeseries dataset
        self._y_dim = 0  # y_dim : the length of each timeseries dataset
        self._value_dimension = 0 # value_dimension : used to determine which dimension data is the values we want
        self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        # if self._training_data is None or self._y_dim==0:
        inputs_timeseries = inputs[1]
        inputs_d3mIndex = inputs[0]
        if not self._fitted:
            return CallResult(None, True, 0)
        if isinstance(inputs_timeseries, np.ndarray):
            X = np.zeros((inputs_timeseries.shape[0], self._y_dim))
        else:
            X = np.zeros((len(inputs_timeseries), self._y_dim))

        for i, series in enumerate(inputs_timeseries):
            if series.shape[1] > 1 and not self._value_found:
                series_output = pd.DataFrame()
                for j in range(series.shape[1]):
                    series_output = pd.concat([series_output,series.iloc[:, j]])
            else:
                series_output = series
            if (series_output.shape[0] < self._y_dim):
                # pad with zeros
                X[i,:series_output.shape[0]] = series_output.iloc[:series_output.shape[0], self._value_dimension]
            else:
                # Truncate or just fit in
                X[i,:] = series_output.iloc[:self._y_dim, self._value_dimension]

        # save the result to DataFrame format
        output_ndarray = self._model.transform(X)
        output_dataFrame = container.DataFrame(output_ndarray)

        # update the original index to be d3mIndex
        output_dataFrame = container.DataFrame(pd.concat([pd.DataFrame(inputs_d3mIndex, columns=['d3mIndex']), pd.DataFrame(output_dataFrame)], axis=1))
        # add d3mIndex metadata
        index_metadata_selector = (mbase.ALL_ELEMENTS, 0)
        index_metadata = {"name": "d3mIndex", "structural_type": str, 'semantic_types': ("https://metadata.datadrivendiscovery.org/types/TabularColumn", "https://metadata.datadrivendiscovery.org/types/PrimaryKey")}
        output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=index_metadata, selector=index_metadata_selector)
        # add other metadata

        if self.hyperparams["generate_metadata"]:
            if type(output_ndarray[0][0]) is np.float64:
                metadata_each_column = {"structural_type": float, 'semantic_types': ("http://schema.org/Float", 'https://metadata.datadrivendiscovery.org/types/Attribute'), }
            else:
                metadata_each_column = {"structural_type": int, 'semantic_types': ("http://schema.org/Integer", 'https://metadata.datadrivendiscovery.org/types/Attribute'), }
            for each_column in range(1, output_dataFrame.shape[1]):
                metadata_selector = (mbase.ALL_ELEMENTS, each_column)
                output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=metadata_each_column, selector=metadata_selector)

            # 2019.4.15: now d3m need to check also inside the query of ((metadata_base.ALL_ELEMENTS,))
            metadata_selector = (mbase.ALL_ELEMENTS, )
            metadata_dimension_columns = {"name":"columns", "semantic_types":("https://metadata.datadrivendiscovery.org/types/TabularColumn",), "length":output_ndarray.shape[1]}
            # d3m require it to be frozen ordered dict
            metadata_dimension_columns = frozendict.FrozenOrderedDict(metadata_dimension_columns)
            metadata_all_elements = {"dimension": metadata_dimension_columns}
            metadata_all_elements = frozendict.FrozenOrderedDict(metadata_all_elements)
            output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=metadata_all_elements, selector=metadata_selector)

            # in the case of further more restricted check, also add metadta query of ()
            metadata_selector = ()
            metadata_dimension_rows = {"name":"rows", "semantic_types":("https://metadata.datadrivendiscovery.org/types/TabularRow",), "length":output_ndarray.shape[0]}
            # d3m require it to be frozen ordered dict
            metadata_dimension_rows = frozendict.FrozenOrderedDict(metadata_dimension_rows)
            metadata_all = {"structural_type":d3m_DataFrame, "semantic_types":  ("https://metadata.datadrivendiscovery.org/types/Table",), "dimension":metadata_dimension_rows, "schema":"https://metadata.datadrivendiscovery.org/schemas/v0/container.json"}
            metadata_all = frozendict.FrozenOrderedDict(metadata_all)
            output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=metadata_all, selector=metadata_selector)

        return CallResult(output_dataFrame, True, None)

    def set_training_data(self, *, inputs: Inputs) -> None:
        if len(inputs) != 2:
            raise InvalidArgumentValueError('Expecting two inputs')

        inputs_timeseries = inputs[1]
        inputs_d3mIndex = inputs[0]
        if len(inputs_timeseries) == 0:
            _logger.info("Warning: Inputs timeseries data to timeseries_featurization primitive's length is 0.")
            return
        # update: now we need to get the whole shape of inputs to process
        lengths = [x.shape[0] for x in inputs_timeseries]
        widths = [x.shape[1] for x in inputs_timeseries]
        # here just take first timeseries dataset to search
        column_name = list(inputs_timeseries[0].columns.values)
        '''
        New things, the previous version only trying to load the fixed columns
        It will cause problems that may load the wrong data
        e.g.: at dataset 66, it will read the "time" data instead of "value"
        So here I added a function to check the name of each column to ensure that we read the correct data
        '''
        for i in range(len(column_name)):
            if 'value' in column_name[i]:
                self._value_found  = True
                self._value_dimension = i

        is_same_length = len(set(lengths)) == 1
        is_same_width = len(set(widths)) == 1
        if not is_same_width:
            _logger.info("Warning: some csv file have different dimensions!")
        if self._value_found :
            if is_same_length:
                self._y_dim = lengths[0]
            else:
                # Truncate all time series to the shortest time series
                self._y_dim = min(lengths)
        else:
            if is_same_length:
                self._y_dim = lengths[0] * widths[0]
            else:
                # Truncate all time series to the shortest time series
                self._y_dim = min(lengths) * min(widths)
        self._x_dim = len(inputs_timeseries)
        self._training_data = np.zeros((self._x_dim, self._y_dim))

        for i, series in enumerate(inputs_timeseries):
            if series.shape[1] > 1 and not self._value_found :
                series_output = pd.DataFrame()
                for each_dimension in range(series.shape[1]):
                    series_output = pd.concat([series_output,series.iloc[:, each_dimension]])
            else:
                series_output = series
            self._training_data[i, :] = series_output.iloc[:self._y_dim, self._value_dimension]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        eps = self.hyperparams['eps']
        n_components = johnson_lindenstrauss_min_dim(n_samples=self._x_dim, eps=eps)
        _logger.info("[INFO] n_components is " + str(n_components))
        if n_components > self._y_dim:
            # Default n_components == 'auto' fails. Need to explicitly assign n_components
            self._model = GaussianRandomProjection(n_components=self._y_dim, random_state=self.random_seed)
        else:
            try:
                self._model = GaussianRandomProjection(eps=eps, random_state=self.random_seed)
                self._model.fit(self._training_data)
            except:
                _logger.info("[Warning] Using given eps value failed, will use default conditions.")
                self._model = GaussianRandomProjection()

        self._model.fit(self._training_data)

        self._fitted = True
        return CallResult(None, has_finished=True)



    def get_params(self) -> Params:
        if self._model:
            return Params(y_dim = self._y_dim,
                          x_dim = self._x_dim,
                          value_found = self._value_found,
                          value_dimension = self._value_dimension,
                          projection_param =  self._model.get_params(),
                          components_ = getattr(self._model, 'components_', None)
                          )
        else:
            return Params({'y_dim': 0, 'projection_param': {}})

    def set_params(self, *, params: Params) -> None:
        self._y_dim = params['y_dim']
        self._x_dim = params['x_dim']
        self._value_found = params['value_found']
        self._value_dimension = params['value_dimension']
        self._model = None
        if params['projection_param']:
            self._model = GaussianRandomProjection()
            self._model.set_params(**params['projection_param'])
            self._model.components_ = params['components_']
            self._fitted = True
        else:
            self._fitted = False

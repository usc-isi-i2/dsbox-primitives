'''
Created on Jan 23, 2018

@author: kyao
'''

import numpy as np
import typing

from d3m.metadata import hyperparams, params
from d3m.container import ndarray
from d3m import container

from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from . import config

Inputs = container.List[container.DataFrame] # this format is for old version of d3m
Outputs = container.DataFrame

class Params(params.Params):
    y_dim: int
    projection_param: typing.Dict
    components_: typing.Optional[np.ndarray]

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

class RandomProjectionTimeSeriesFeaturization(FeaturizationLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    classdocs
    '''

    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox.timeseries_featurization.random_projection",
        "version": config.VERSION,
        "name": "DSBox random projection timeseries featurization ",
        "description": "A simple timeseries featurization using random projection",
        "python_path": "d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization",
        "primitive_family": "FEATURE_EXTRACTION",
        "algorithm_types": [ "RANDOM_PROJECTION" ],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
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
        self._x_dim = 0  # x_dim : the amount of timeseries dataset
        self._y_dim = 0  # y_dim : the length of each timeseries dataset
        self._value_dimension = 0 # value_dimension : used to determine which dimension data is the values we want
        self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # if self._training_data is None or self._y_dim==0:
        if not self._fitted:
            return CallResult(None, True, 0)
        if isinstance(inputs, np.ndarray):
            X = np.zeros((inputs.shape[0], self._y_dim))
        else:
            X = np.zeros((len(inputs), self._y_dim))


        for i, series in enumerate(inputs):
            if (series.shape[0] < self._y_dim):
                # pad with zeros
                X[i,:series.shape[0]] = series.iloc[:series.shape[0], self._value_dimension]
            else:
                # Truncate or just fit in
                X[i,:] = series.iloc[:self._y_dim, self._value_dimension]
        return CallResult(self._model.transform(X), True, 1)

    def set_training_data(self, *, inputs: Inputs) -> None:
        if len(inputs) == 0:
            return
        lengths = [x.shape[0] for x in inputs]
        is_same_length = len(set(lengths)) == 1
        if is_same_length:
            self._y_dim = lengths[0]
        else:
            # Truncate all time series to the shortest time series
            self._y_dim = min(lengths)
        self._x_dim = len(inputs)
        self._training_data = np.zeros((self._x_dim, self._y_dim))
        '''
        New things, the previous version only trying to load the fixed columns
        It will cause problems that may load the wrong data
        e.g.: at dataset 66, it will read the "time" data instead of "value"
        So here I added a function to check the name of each column to ensure that we read the correct data
        '''
        # here just take first timeseries dataset

        column_name = list(inputs[0].columns.values)
        for i in range(len(column_name)):
            if 'value' in column_name[i]:
                self._value_dimension = i

        for i, series in enumerate(inputs):
            '''
            # For testing purpose of the input data
            print("~~~~~",i,"~~~~~~")
            print(series.iloc[:self._y_dim, 1])
            '''
            self._training_data[i, :] = series.iloc[:self._y_dim, self._value_dimension]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        eps = self.hyperparams['eps']
        n_components = johnson_lindenstrauss_min_dim(n_samples=self._x_dim, eps=eps)

        print("n_components is", n_components)
       
        if n_components > self._x_dim:
            # Default n_components == 'auto' fails. Need to explicitly assign n_components
            self._model = GaussianRandomProjection(n_components=self._x_dim, random_state=self.random_seed)
        else:
            self._model = GaussianRandomProjection(eps=eps, random_state=self.random_seed)
        self._model.fit(self._training_data)
        self._fitted = True

    def get_params(self) -> Params:
        if self._model:
            return Params(y_dim = self._y_dim,
                          projection_param =  self._model.get_params(),
                          components_ = getattr(self._model, 'components_', None))
        else:
            return Params({'y_dim': 0, 'projection_param': {}})

    def set_params(self, *, params: Params) -> None:
        self._y_dim = params['y_dim']
        self._model = None
        if params['projection_param']:
            self._model = GaussianRandomProjection()
            self._model.set_params(**params['projection_param'])
            self._model.components_ = params['components_']
            self._fitted = True
        else:
            self._fitted = False

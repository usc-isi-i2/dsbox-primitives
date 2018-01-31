'''
Created on Jan 23, 2018

@author: kyao
'''

import numpy as np
import typing

from d3m_metadata import container, hyperparams, params
from d3m_metadata.metadata import PrimitiveMetadata

from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from primitive_interfaces.featurization import FeaturizationPrimitiveBase, CallResult

from . import config

Inputs = container.List[container.DataFrame]
Outputs = container.ndarray

class Params(params.Params):
    y_dim: int
    projection_param: dict

class Hyperparams(hyperparams.Hyperparams):
    eps = hyperparams.Uniform(lower=0.1, upper=0.5, default=0.2)

class RandomProjectionTimeSeriesFeaturization(FeaturizationPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    classdocs
    '''

    metadata = PrimitiveMetadata({
        "id": "dsbox.timeseries_featurization.random_projection",
        "version": config.VERSION,
        "name": "DSBox random projection timeseries featurization ",
        "description": "A simple timeseries featurization using random projection",
        "python_path": "d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization",
        "primitive_family": "FEATURE_EXTRACTION",
        "algorithm_types": [ "RANDOM_PROJECTION" ],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "uris": [ 'git+https://github.com/usc-isi-i2/dsbox-featurizer' ]
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

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, 
                 docker_containers: typing.Dict[str, str] = None) -> None:
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers
        self._model = None
        self._training_data = None
        self._x_dim = 0
        self._y_dim = 0

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._training_data is None or self._y_dim==0:
            return CallResult(None, True, 0)
        if isinstance(inputs, np.ndarray): 
            X = np.zeros((inputs.shape[0], self._y_dim))
        else:
            X = np.zeros((len(inputs), self._y_dim))
        for i, series in enumerate(inputs):
            X[i,:] = series.iloc[:self._y_dim, 0]
        return CallResult(self._model.transform(X), True, 1)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
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
        for i, series in enumerate(inputs):
            self._training_data[i, :] = series.iloc[:self._y_dim, 0]
            
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        eps = self.hyperparams['eps']
        n_components = johnson_lindenstrauss_min_dim(n_samples=self._x_dim, eps=eps)
        if n_components > self._x_dim: 
            self._model = GaussianRandomProjection(n_components=self._x_dim)
        else:
            self._model = GaussianRandomProjection(eps=eps)
        self._model.fit(self._training_data)

    def get_params(self) -> Params:
        if self._model:
            return Params(y_dim=self._y_dim,
                          projection_param=self._model.get_params())
        else:
            return Params({'y_dim': 0, 'projection_param': {}})

    def set_params(self, *, params: Params) -> None:
        self._y_dim = params['y_dim']
        self._model = None
        if params['projection_param']:
            self._model = GaussianRandomProjection()
            self._model.set_params(**params['projection_param'])
        

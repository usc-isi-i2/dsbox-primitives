from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from primitive_interfaces.base import CallResult

from d3m_metadata import container, hyperparams, metadata
from d3m_metadata.container import DataFrame, List, ndarray
import typing

from .relation_matrix_all import get_relation_matrix
from .helper import Aggregator
from .relationMatrix2foreignKey import relationMat2foreignKey


Inputs = typing.Union[List[DataFrame], List[str]]   # tables, their names, master table name and column
Outputs = DataFrame


class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """

    pass


class MultiTableFeaturization(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    __author__ = 'USC ISI'
    metadata = metadata.PrimitiveMetadata({
        'id': 'dsbox-multi-table-featurization-aggregation',
        'version': 'v1.0',        
        'name': "DSBox Multiple Table Featurizer Aggregation",
        'description': 'Generate a featurized table from multiple-table dataset using aggregation',
        'python_path': 'd3m.primitives.dsbox.multiTableFeaturization',
        'primitive_family': metadata.PrimitiveFamily.FEATURE_EXTRACTION, # TODO: change this
        'algorithm_types': [metadata.PrimitiveAlgorithmType.DATA_PROFILING, ],  # TODO: change this
        'keywords': ['multiple table'],
        'source': {
            'name': __author__,
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                # 'https://github.com/usc-isi-i2/dsbox-profiling/blob/master/dsbox/datapreprocessing/profiler/data_profile.py',
                # 'https://github.com/usc-isi-i2/dsbox-profiling.git',
                ],
        },
            # The same path the primitive is registered with entry points in setup.py.
        'installation': [{
                'type': metadata.PrimitiveInstallationType.PIP,
                'package_uri': 'https://github.com/usc-isi-i2/dsbox-profiling.git' # TODO: change this
            }],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        "precondition": [],
        "hyperparms_to_tune": []
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, 
                 docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:

        # All primitives must define these attributes
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers

        # All other attributes must be private with leading underscore    
        self._has_finished = False
        self._iterations_done = False
        self._verbose = hyperparams['verbose'] if hyperparams else 0

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        data = inputs[0]
        names = inputs[1][:-1]
        master_table_name = inputs[1][-1]
        # TODO: format names, make sure it has to be like xx.csv
        # for example, if original names is like [0,1,2], make it to be [0.csv, 1.csv, 2.csv]

        # step 1: get relation matrix 
        relation_matrix = get_relation_matrix(data, names)

        # step 2: get prime key - foreign key relationship
        relations = relationMat2foreignKey(data, names, relation_matrix)
        # print (relations) # to see if the relations are correct

        # step 3: featurization
        aggregator = Aggregator(relations, data, names)
        big_table = aggregator.forward(master_table_name)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(big_table, self._has_finished, self._iterations_done)



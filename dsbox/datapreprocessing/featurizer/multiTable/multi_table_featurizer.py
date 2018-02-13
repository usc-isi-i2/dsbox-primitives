from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from primitive_interfaces.base import CallResult

from d3m_metadata import container, hyperparams, metadata
from d3m_metadata.container import DataFrame, List, ndarray
import typing
import math
import stopit #  type: ignore

from .relation_matrix_all import get_relation_matrix
from .helper import Aggregator
from .relationMatrix2foreignKey import relationMat2foreignKey
from . import config
from d3m_metadata.hyperparams import UniformInt

Inputs = typing.Union[List[DataFrame], List[str]]   # tables, their names, master table name and column
Outputs = DataFrame


class Hyperparams(hyperparams.Hyperparams):
    verbose = UniformInt(lower=0, upper=1, default=0)


class MultiTableFeaturization(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    __author__ = 'USC ISI'
    metadata = metadata.PrimitiveMetadata({
        'id': 'dsbox-multi-table-featurization-aggregation',
        'version': config.VERSION,       
        'name': "DSBox Multiple Table Featurizer Aggregation",
        'description': 'Generate a featurized table from multiple-table dataset using aggregation',
        'python_path': 'd3m.primitives.dsbox.MultiTableFeaturization',
        'primitive_family': metadata.PrimitiveFamily.FEATURE_EXTRACTION,
        'algorithm_types': [metadata.PrimitiveAlgorithmType.RELATIONAL_DATA_MINING, ],
        'keywords': ['multiple table'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            'uris': [ config.REPOSITORY ]
            },
            # The same path the primitive is registered with entry points in setup.py.
        'installation': [ config.INSTALLATION ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
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
        
        if (timeout is None):
            big_table = self._core(inputs)
            self._has_finished = True
            self._iterations_done = True
            return CallResult(big_table, self._has_finished, self._iterations_done)
        else:
            # setup the timeout
            with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
                assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING

                # core computations
                big_table = self._core(inputs)

            if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
                self._has_finished = True
                self._iterations_done = True
                return CallResult(big_table, self._has_finished, self._iterations_done)
            elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
                self._has_finished = False
                self._iterations_done = False
                return CallResult(None, self._has_finished, self._iterations_done)



    def _core(self, inputs) -> Outputs:
        """
        core calculations
        """
        data = inputs[0]
        names = inputs[1][:-1]
        master_table_name = inputs[1][-1]
        # TODO: format names, make sure it has to be like xx.csv
        # for example, if original names is like [0,1,2], make it to be [0.csv, 1.csv, 2.csv]

        # step 1: get relation matrix 
        relation_matrix = get_relation_matrix(data, names)
        if self.verbose > 0:
            relation_matrix.to_csv("./relation_matrix.csv", index=False)
        # step 2: get prime key - foreign key relationship
        relations = relationMat2foreignKey(data, names, relation_matrix)
        # print (relations) # to see if the relations are correct

        # step 3: featurization
        aggregator = Aggregator(relations, data, names)
        big_table = aggregator.forward(master_table_name)

        return big_table

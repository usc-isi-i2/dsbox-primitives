import logging
from d3m import container
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams
from d3m.base import utils as d3m_utils
from d3m.metadata.base import ALL_ELEMENTS
from . import config
import wikifier
import frozendict
import pandas as pd

Inputs = container.Dataset
Outputs = container.Dataset
_logger = logging.getLogger(__name__)


class WikifierHyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    specific_q_nodes = hyperparams.Hyperparameter[list](
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="specified Q nodes used for searching, if not given, will try to find one",
    )

class Wikifier(TransformerPrimitiveBase[Inputs, Outputs, WikifierHyperparams]):
    '''
    A primitive that takes a list of datamart dataset and choose 1 or a few best dataframe and perform join, return an accessible d3m.dataframe for further processing
    '''
    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "wikidata-wikifier",
        "version": config.VERSION,
        "name": "wikidata wikifier",
        "python_path": "d3m.primitives.data_augmentation.wikifier.DSBOX",
        "description": "find the corresponding Q nodes values of given columns",
        "primitive_family": "DATA_AUGMENTATION",
        "algorithm_types": ["APPROXIMATE_DATA_AUGMENTATION"],  # fix me!
        "keywords": ["data augmentation", "wikidata", "wikifier"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "installation": [config.INSTALLATION],
        # 'precondition': [],
        # 'hyperparams_to_tune': []

    })

    def __init__(self, *, hyperparams: WikifierHyperparams)-> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = False
        self._target_columns = set()
        self._res_id = None
        self._inputs_df = None
        self._inputs_ds = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        self._inputs_ds = inputs.copy()
        self._res_id, self._inputs_df = d3m_utils.get_tabular_resource(dataset=inputs, resource_id=None)

        if len(self.hyperparams['use_columns']) > 0:
            for each_column in self.hyperparams['use_columns']:
                if each_column < 0 or each_column >= self._inputs_df.shape[1]:
                    _logger.warning("Out-of-bound column number" + str(each_column) + "detected! Will skip this column.")
                else:
                    self._target_columns.add(each_column)

        if len(self.hyperparams['exclude_columns']) > 0 and len(self.hyperparams['use_columns']) > 0:
            _logger.warning("Both exclude columns and use_columns were given, will ignore exclude column settings")

        elif len(self.hyperparams['exclude_columns']) > 0:
            self._target_columns = set(range(self._inputs_df.shape[1]))
            for each_column in self.hyperparams['exclude_columns']:
                if each_column < 0 or each_column >= self._inputs_df.shape[1]:
                    _logger.warning(
                        "Out-of-bound column number" + str(each_column) + "detected! Will skip this column.")
                else:
                    self._target_columns.remove(each_column)
        elif len(self.hyperparams['use_columns']) == 0 and len(self.hyperparams['exclude_columns']) == 0:
            self._target_columns = set(range(self._inputs_df.shape[1]))

        temp = self._target_columns.copy()
        for each in self._target_columns:
            each_column_semantic_type = self._inputs_ds.metadata.query((self._res_id, ALL_ELEMENTS, each))['semantic_types']
            if 'http://schema.org/Integer' in each_column_semantic_type or 'http://schema.org/Float' in each_column_semantic_type:
                temp.remove(each)
        self._target_columns = temp
        _logger.info(self._target_columns)

        _logger.info("final target columns are:" + str(temp))
        result = self._produce_for_d3m_dataset()
        self._has_finished = True
        self._iterations_done = True
        return CallResult(result, True, 1)

    def _produce_for_d3m_dataframe(self):
        """
        function used to produce for input type is d3m.container.dataFrame, use for d3m dataFrame format only
        """
        # may not need to use for now
        pass

    def _produce_for_d3m_dataset(self) -> Outputs:
        """
        function used to produce for input type is d3m.container.Dataset, use for d3m Dataset format only
        """
        output_ds = self._inputs_ds
        if len(self.hyperparams['specific_q_nodes']) > 0:
            specific_q_nodes = self.hyperparams['specific_q_nodes']
        else:
            specific_q_nodes = None

        wikifier_res = wikifier.produce(pd.DataFrame(self._inputs_df), list(self._target_columns), specific_q_nodes)
        output_ds[self._res_id] = d3m_DataFrame(wikifier_res, generate_metadata=False)
        # update metadata on column length
        selector = (self._res_id, ALL_ELEMENTS)
        old_meta = dict(output_ds.metadata.query(selector))
        old_meta_dimension = dict(old_meta['dimension'])
        old_column_length = old_meta_dimension['length']
        old_meta_dimension['length'] = wikifier_res.shape[1]
        old_meta['dimension'] = frozendict.FrozenOrderedDict(old_meta_dimension)
        new_meta = frozendict.FrozenOrderedDict(old_meta)
        output_ds.metadata = output_ds.metadata.update(selector, new_meta)

        # update each column's metadata
        for i in range(old_column_length, wikifier_res.shape[1]):
            selector = (self._res_id, ALL_ELEMENTS, i)
            metadata = {"name": wikifier_res.columns[i],
                        "structural_type": str,
                        'semantic_types': (
                            "http://schema.org/Text",
                            "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                            "https://metadata.datadrivendiscovery.org/types/Attribute",
                            "http://wikidata.org/qnode"
                        )}
            output_ds.metadata = output_ds.metadata.update(selector, metadata)

        return output_ds

import logging
from typing import List, Dict
from d3m.metadata.base import DataMetadata
from d3m import container, exceptions
import d3m.metadata.base as mbase
# from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase
# changed class to fit in devel branch of d3m (2019-1-17)
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
import common_primitives.utils as common_utils
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.base import CallResult


from . import config

#__all__ = ('Labler',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    model: Dict
    s_cols: List[object]
    fitted: bool


class LablerHyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. "
                    "If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns,"
                    " or should only parsed columns be returned? "
                    "This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. "
                    "Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. "
                    "Applicable only if \"return_result\" is set to \"new\".",
    )


class Labler(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, LablerHyperparams]):
    """
    A primitive which encode all categorical values into integers. This primitive can
    handle values not seen during training.
    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-multi-table-feature-labler",
        "version": config.VERSION,
        "name": "DSBox feature labeler",
        "description": "A simple primitive that labels all string based categorical columns",
        "python_path": "d3m.primitives.data_cleaning.label_encoder.DSBOX",
        "primitive_family": "DATA_CLEANING",
        "algorithm_types": ["DATA_NORMALIZATION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["NORMALIZATION", "Labeler"],
        "installation": [config.INSTALLATION],
        "precondition": ["NO_MISSING_VALUES", "CATEGORICAL_VALUES"],

    })

    def __init__(self, *, hyperparams: LablerHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._training_data = None
        self._fitted = False
        self._s_cols: List = []
        self._model: Dict = {}
        self._has_finished = False
        self.logger = logging.getLogger(__name__)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_data = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        self._fitted = True
        categorical_attributes = DataMetadata.list_columns_with_semantic_types(
            self=self._training_data.metadata,
            semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                "https://metadata.datadrivendiscovery.org/types/CategoricalData"
                ]
            )

        all_attributes = DataMetadata.list_columns_with_semantic_types(
            self=self._training_data.metadata,
            semantic_types=["https://metadata.datadrivendiscovery.org/types/Attribute"]
            )

        self._s_cols = container.List(set(all_attributes).intersection(categorical_attributes))
        self.logger.debug("%d of categorical attributes found." % (len(self._s_cols)))

        if len(self._s_cols) > 0:
            # temp_model = defaultdict(LabelEncoder)
            # self._training_data.iloc[:, self._s_cols].apply(lambda x: temp_model[x.name].fit(x))
            # self._model = dict(temp_model)
            self._model = {}
            for col_index in self._s_cols:
                self._model[col_index] = self._training_data.iloc[:, col_index].dropna().unique()

        return CallResult(None, has_finished=True)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError('Labeler not fitted')

        if len(self._s_cols) == 0:
            # No categorical columns. Nothing to do.
            return CallResult(inputs, True)

        # Generate label encoding
        columns = []
        for col_index in self._s_cols:
            size = self._model[col_index].size
            mapping = {x: i for i, x in enumerate(self._model[col_index])}
            columns.append(inputs.iloc[:, col_index].apply(lambda x: mapping[x] if x in mapping else size))

        # insert encoded columns
        outputs = inputs.copy()
        for col, index in enumerate(self._s_cols):
            outputs.iloc[:, index] = columns[col]

        lookup = {
            "int": ('http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute')
        }

        for index in self._s_cols:
            old_metadata = dict(outputs.metadata.query((mbase.ALL_ELEMENTS, index)))
            old_metadata["semantic_types"] = lookup["int"]
            old_metadata["structural_type"] = type(10)
            outputs.metadata = outputs.metadata.update((mbase.ALL_ELEMENTS, index), old_metadata)

        self._has_finished = True
        return CallResult(outputs, self._has_finished)

        # remove the columns that appeared in produce method but were not in fitted data
        # drop_names = set(outputs.columns[self._s_cols]).difference(set(self._model.keys()))
        # drop_indices = map(lambda a: outputs.columns.get_loc(a), drop_names)
        # drop_indices = sorted(drop_indices)
        # outputs = common_utils.remove_columns(outputs, drop_indices)

        # # sanity check and report the results
        # if outputs.shape[0] == inputs.shape[0] and \
        #    outputs.shape[1] == inputs.shape[1] - len(drop_names):
        #     self._has_finished = True
        #     self._iterations_done = True
        #     # print("output:",outputs.head(5))
        #     return CallResult(d3m_DataFrame(outputs), self._has_finished, self._iterations_done)
        # else:
        #     return CallResult(inputs, self._has_finished, self._iterations_done)

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: LablerHyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(
            inputs_metadata, use_columns=hyperparams['use_columns'], exclude_columns=hyperparams['exclude_columns'],
            can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: mbase.DataMetadata, column_index: int, hyperparams: LablerHyperparams) -> bool:
        column_metadata = inputs_metadata.query((mbase.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    def get_params(self) -> Params:
        return Params(model=self._model, s_cols=self._s_cols, fitted=self._fitted)

    def set_params(self, *, params: Params) -> None:
        self._s_cols = params['s_cols']
        self._model = params['model']
        self._fitted = params['fitted']

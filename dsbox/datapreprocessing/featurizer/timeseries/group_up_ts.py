
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
# from . import missing_value_pred as mvp
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import stopit
import math
import typing
import logging

from d3m import container
from d3m.metadata import hyperparams, params
from d3m.container.list import List
from d3m.metadata.hyperparams import UniformBool
import d3m.metadata.base as mbase
import common_primitives.utils as utils

from . import config



_logger = logging.getLogger(__name__)

Input = container.DataFrame
Output = container.List

#output : a list, [dataframe, a dictionary for data]




class GroupUpHyperparameter(hyperparams.Hyperparams):
    verbose = UniformBool(default=False,
                          semantic_types=['http://schema.org/Boolean',
                                          'https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    # exclude_columns = hyperparams.Set(
    #     elements=hyperparams.Hyperparameter[int](-1),
    #     default=(),
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    # )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    # use_semantic_types = hyperparams.UniformBool(
    #     default=False,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    # )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )

class GroupUpByTimeSeries(TransformerPrimitiveBase[Input, Output,GroupUpHyperparameter]):
    """
    Impute the missing value by greedy search of the combinations of standalone simple imputation method.

    Parameters:
    ----------
    verbose: bool
        Control the verbosity

    Attributes:
    ----------
    imputation_strategies: list of string,
        each is a standalone simple imputation method

    best_imputation: dict. key: column name; value: trained imputation method (parameters)
            which is one of the imputation_strategies

    model: a sklearn machine learning class
        The machine learning model that will be used to evaluate the imputation strategies

    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        # Required
        "id": 'd2bd4f3b-fd43-47c2-8ab7-f9fce691e2b2',
        "version": config.VERSION,
        "name": "DSBox Group Up by Timeseries Primitive",
        "description": "Group up timeseries data by same name of timeseries files",

        "python_path": "d3m.primitives.data_transformation.group_up_by_timeseries.DSBOX",
        "primitive_family": "DATA_TRANSFORMATION",
        "algorithm_types": ["NUMERICAL_METHOD"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        # Automatically generated
        # "primitive_code"
        # "original_python_path"
        # "schema"
        # "structural_type"
        # Optional
        "keywords": ["Transform", "Timeseries", "Aggregate"],
        "installation": [config.INSTALLATION],
        "precondition":["NO_CATEGORICAL_VALUES"]
    })

    def __init__(self, *, hyperparams: GroupUpHyperparameter) -> None:
        super().__init__(hyperparams=hyperparams)
        # All primitives must define these attributes
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = False
        # self._verbose = hyperparams['verbose'] if hyperparams else False


    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> CallResult[Output]:
        """
        precond: run fit() before

        Parameters:
        ----------
        data: pandas dataframe
        """

        if (timeout is None):
            timeout = 2**31-1

        if isinstance(inputs, pd.DataFrame):
            data = inputs.copy()
        else:
            data = inputs[0].copy()
        # setup the timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mrg:
            assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING
            # query for timeseries data
            elements_amount = inputs.metadata.query((mbase.ALL_ELEMENTS,))['dimension']['length']
            d3mIndex_output = np.asarray(inputs.index.tolist())

            # traverse each selector to check where is the image file
            timeseries_index = -1
            timeseries_file = []
            for selector_index in range(elements_amount):
                each_selector = inputs.metadata.query((mbase.ALL_ELEMENTS,selector_index))
                mime_types_found = False
                if 'mime_types' in each_selector:
                    mime_types = (each_selector['mime_types'])
                    mime_types_found = True
                elif 'media_types' in each_selector:
                    mime_types = (each_selector['media_types'])
                    mime_types_found = True
                if mime_types_found:
                    timeseries_file.append(selector_index)
            if len(timeseries_file) == 1:
                data_grouped = data.groupby(data.columns[timeseries_file[0]])
            elif len(timeseries_file) == 0:
                _logger.info("No timeseries_file specified")
                return CallResult(None, self._has_finished, self._iterations_done)
            else:
                _logger.info("Cannot handle more than 1 timeseries_file")
                return CallResult(None, self._has_finished, self._iterations_done)

            # Start with storing the result
            # create a List
            TS_mapper = {}
            for k, v in data_grouped:
                TS_mapper[k] = v.index  # should be a list containing indexes mapping to ts file 
                _logger.info("Timeseries data Group up finished.")
        value = None
        if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
            self._has_finished = True
            self._iterations_done = True
            value = List()
            value.append(inputs)
            value.append(TS_mapper)
        elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
            _logger.warn('Produce timed out')
            self._has_finished = False
            self._iterations_done = False

        return CallResult(value, self._has_finished, self._iterations_done)


    @classmethod
    def _get_columns_to_fit(cls, inputs: Input, hyperparams: GroupUpHyperparameter):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_columns'],
                                                                             exclude_columns=hyperparams['exclude_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce


    @classmethod
    def _can_produce_column(cls, inputs_metadata: mbase.DataMetadata, column_index: int, hyperparams: GroupUpHyperparameter) -> bool:
        column_metadata = inputs_metadata.query((mbase.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False



if __name__ == "__main__":
    pass



'''
Horizontal concatenation of multiple dataframes
'''
import logging
import typing

import pandas as pd

from d3m import container
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata.base import ALL_ELEMENTS

from dsbox.datapreprocessing.cleaner import config

__all__ = ('HorizontalConcat',)

_logger = logging.getLogger(__name__)

Inputs = container.List
Outputs = container.DataFrame

class HorizontalConcatHyperparams(hyperparams.Hyperparams):
    """Hyperaparams for the HorizontalConcatPrimitive"""
    use_index = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Use primary index columns in all DataFrames (if they exist) to match rows in proper order. Otherwise, concatination happens on the order of rows in input DataFrames.",
    )
    keep_first_index_only = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="When more than input DataFrames have primary index columns, keep the index from the first DataFrame and remove all other indices from the result."
                    " When \"use_index\" is \"True\", all other index columns are redundant because they are equal to the first ones (assuming equal metadata).",
    )
    auto_rename_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Automatically rename column names, if concatenation results in duplicate column names."
    )


class HorizontalConcat(TransformerPrimitiveBase[Inputs, Outputs, HorizontalConcatHyperparams]):
    """
        A primitive which concat a list of dataframe to a single dataframe horizontally,
        and it will also set metatdata for prediction,
        we assume that inputs has same length
    """

    __author__ = "USC ISI"
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "dsbox-horizontal-concat",
        "version": config.VERSION,
        "name": "DSBox horizontal concat",
        "description": "horizontally concat a list of dataframe",
        "python_path": "d3m.primitives.data_transformation.horizontal_concat.DSBOX",
        "primitive_family": "DATA_TRANSFORMATION",
        "algorithm_types": ["DATA_CONVERSION"],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["concat", "horizontal"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: HorizontalConcatHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        use_index = self.hyperparams['use_index']
        keep_first_index_only = self.hyperparams['keep_first_index_only']
        auto_rename_columns = self.hyperparams['auto_rename_columns']

        if len(inputs) == 0:
            raise ValueError("Input list must have at least one element.")
        if len(inputs) == 1:
            return CallResult(inputs[0])

        first = inputs[0]
        all_dfs = [first]
        for other in inputs[1:]:
            first.metadata._check_same_number_of_samples(other.metadata)

        first_indices = first.metadata.get_index_columns()
        column_names = set(first.columns)
        _logger.debug(f'Column names from first dataframe: {column_names}')
        for index, other in enumerate(inputs[1:]):
            if auto_rename_columns:
                rename = {}
                _logger.debug(f'Column names from dataframe {index+1}: {set(other.columns)}')
                for name in other.columns:
                    if name == 'd3mIndex':
                        continue
                    if name in column_names:
                        new_name = self._new_name(name, column_names)
                        rename[name] = new_name
                        column_names.add(new_name)
                    else:
                        column_names.add(name)
                if rename:
                    _logger.debug(f'Renaming dataframe number {index+1}:')
                    if _logger.getEffectiveLevel() <= 10:
                        for key, value in rename.items():
                            _logger.debug(f'  {key} -> {value}')
                    other = self._rename_columns(other, rename)
            other_indices = other.metadata.get_index_columns()
            if first_indices and other_indices:
                if use_index:
                    old_other_metadata = other.metadata
                    other = self._sort_other_indices(first, other, first_indices, other_indices)
                    # TODO: Reorder metadata rows as well.
                    #       This should be relatively easy because we can just modify
                    #       "other.metadata._current_metadata.metadata" map.
                    other.metadata = old_other_metadata

            # Removing second primary key columns.
            if keep_first_index_only:
                other = other.remove_columns(other_indices)
            all_dfs.append(other)
        return CallResult(self._concat_dataframes(all_dfs))

    @staticmethod
    def _rename_columns(dataframe: container.DataFrame, mapping: typing.Dict[str, str]) -> container.DataFrame:
        '''Rename DataFrame column names, and corresponding metadata column names'''
        dataframe = dataframe.rename(columns=mapping)
        for index in range(dataframe.metadata.query((ALL_ELEMENTS,))['dimension']['length']):
            column_metadata = dataframe.metadata.query((ALL_ELEMENTS, index))
            if column_metadata['name'] in mapping:
                column_metadata = dict(column_metadata)
                column_metadata['name'] = mapping[column_metadata['name']]
                dataframe.metadata = dataframe.metadata.update((ALL_ELEMENTS, index), column_metadata)
        return dataframe

    @staticmethod
    def _new_name(name: str, existing_names: set):
        prefix = 'hc'
        new_name = name
        while new_name in existing_names:
            parts = new_name.split('_')
            if parts[-1][:2] == prefix and parts[-1][2:].isdigit():
                parts[-1] = prefix + str(int(parts[-1][2:]) + 1)
            else:
                parts.append(prefix + '1')
            new_name = '_'.join(parts)
        return new_name

    @staticmethod
    def _concat_dataframes(inputs: Inputs) -> Outputs:

        # Do not use Pandas index, just concat based on row position
        outputs = pd.concat([df.reset_index(drop=True) for df in inputs], axis=1)

        outputs.metadata = inputs[0].metadata
        for other in inputs[1:]:
            outputs.metadata = outputs.metadata.append_columns(other.metadata)
        return outputs

    @staticmethod
    def _sort_other_indices(first: container.DataFrame, right: container.DataFrame, indices: typing.Sequence[int],
                            right_indices: typing.Sequence[int]) -> container.DataFrame:
        # Based on DataFrame._sort_right_indices()

        # We try to handle different cases.

        # We do not do anything special. We assume both indices are the same.
        if len(indices) == 1 and len(right_indices) == 1:
            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right.iloc[:, right_indices[0]]).reindex(first.iloc[:, indices[0]]).reset_index(drop=True)

        index_names = [first.metadata.query_column(index).get('name', None) for index in indices]
        right_index_names = [right.metadata.query_column(right_index).get('name', None) for right_index in right_indices]

        index_series = [first.iloc[:, index] for index in indices]
        right_index_series = [right.iloc[:, right_index] for right_index in right_indices]

        # Number match, names match, order match, things look good.
        if index_names == right_index_names:
            # We know the length is larger than 1 because otherwise the first case would match.
            assert len(indices) > 1
            assert len(indices) == len(right_indices)

            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right_index_series).reindex(index_series).reset_index(drop=True)

        sorted_index_names = sorted(index_names)
        sorted_right_index_names = sorted(right_index_names)

        # Number and names match, but not the order.
        if sorted_index_names == sorted_right_index_names:
            # We know the length is larger than 1 because otherwise the first case would match.
            assert len(indices) > 1
            assert len(indices) == len(right_indices)

            # We sort index series to be in the sorted order based on index names.
            index_series = [s for _, s in sorted(zip(index_names, index_series), key=lambda pair: pair[0])]
            right_index_series = [s for _, s in sorted(zip(right_index_names, right_index_series), key=lambda pair: pair[0])]

            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right_index_series).reindex(index_series).reset_index(drop=True)

        if len(index_series) == len(right_index_series):
            # We know the length is larger than 1 because otherwise the first case would match.
            assert len(indices) > 1

            _logger.warning("Primary indices both on left and right not have same names, but they do match in number.")

            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right_index_series).reindex(index_series).reset_index(drop=True)

        # It might be that there are duplicate columns on either or even both sides,
        # but that should be resolved by adding a primitive to remove duplicate columns first.
        raise ValueError("Left and right primary indices do not match in number.")


    # def produce(self, *, inputs1: Inputs, inputs2: Inputs,
    #             timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
    #     # need to rename inputs1.columns and inputs2.columns name
    #     if self.hyperparams["column_name"] == 0:
    #         left = inputs1.rename(columns={inputs1.columns[-1]: "0"})
    #     else:
    #         left = copy(inputs1)
    #     right = inputs2.rename(
    #         columns={inputs2.columns[-1]: str(self.hyperparams["column_name"]+1)})
    #     new_df = common_utils.horizontal_concat(left, right)

    #     for i, column in enumerate(new_df.columns):
    #         column_metadata = dict(new_df.metadata.query((ALL_ELEMENTS, i)))
    #         if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in column_metadata["semantic_types"]:
    #             column_metadata["semantic_types"] = self.hyperparams["to_semantic_types"]
    #             new_df.metadata = new_df.metadata.update(
    #                 (ALL_ELEMENTS, i), column_metadata)
    #     return CallResult(new_df)

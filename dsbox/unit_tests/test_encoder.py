import unittest
from typing import Dict, List, NamedTuple

import pandas as pd
import numpy as np
from d3m.container.pandas import DataFrame
from d3m.metadata import base as metadata_base
from d3m import index
from dsbox.datapreprocessing.cleaner.encoder import Encoder

class Column(NamedTuple):
    name: str
    data: list
    semantic_types: List[str]

class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.hyperparams_class = Encoder.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self.categorical = "https://metadata.datadrivendiscovery.org/types/CategoricalData"
        self.ordinal = "https://metadata.datadrivendiscovery.org/types/OrdinalData"
        self.attribute = "https://metadata.datadrivendiscovery.org/types/Attribute"

        # Includes int and str categories
        self.categorical_data = self._make_df([
            Column('int_cat', [1, 1, 2, 1], [self.attribute, self.categorical]),
            Column('str_cat', ['a', 'b', 'b', 'a'], [self.attribute, self.categorical]),
            Column('num', [3.4, 4.5, 1.1, 8.2], [self.attribute])
        ])
        # Same as `categorical_data` but with unseen values
        self.categorical_data2 = self._make_df([
            Column('int_cat', [1, 3, 2, 1], [self.attribute, self.categorical]),
            Column('str_cat', ['a', 'b', 'b', 'c'], [self.attribute, self.categorical]),
            Column('num', [7.4, 1.5, 3.1, 5.2], [self.attribute])
        ])
        # Contains categorical and ordinal values that are floats
        self.categorical_ordinal_float_data = self._make_df([
            Column('float_cat', [1.0, 2.0, 2.0, 3.4], [self.attribute, self.categorical]),
            Column('float_ord', [4.0, 4.0, 5.0, 6.1], [self.attribute, self.ordinal]),
            Column('num', [1, 2, 3, 4], [self.attribute])
        ])
        # Contains an ordinal column
        self.ordinal_data = self._make_df([
            Column('ord', [1, 1, 2, 1], [self.attribute, self.ordinal]),
            Column('num', [3.4, 4.5, 1.1, 8.2], [self.attribute])
        ])
        # All numeric i.e. no categories.
        self.numeric_data = self._make_df([
            Column('intcol', [1, 1, 2, 1], [self.attribute]),
            Column('floatcol', [3.4, 4.5, 1.1, 8.2], [self.attribute])
        ])
        # Includes NaNs in every column
        self.categorical_with_nans_data = self._make_df([
            Column('cat', [np.nan, 5, 4, 4, np.nan], [self.attribute, self.categorical]),
            Column('num', [1.6, 8.2, np.nan, 9.0, np.nan], [self.attribute])
        ])
        # Includes values that would break on the column renaming logic
        # in the encoder if it was implemented incorrectly.
        self.str_edge_case_data = self._make_df([
            Column('num', [10, 11, 12, 12, 13], [self.attribute]),
            Column('str_cat', ['a', 'a.0', 'a', 'a.0', 'b'], [self.attribute, self.categorical])
        ])
    
    def test_can_encode_categorical(self):
        """
        Ensures categorical attributes are encoded.
        """
        after_onehot = self._fit_produce(self.categorical_data)

        self.assertEqual(after_onehot.shape, (4, 7))

        self.assertEqual(after_onehot['int_cat_1'].sum(), 3)
        self.assertEqual(after_onehot['int_cat_2'].sum(), 1)
        self.assertEqual(after_onehot['int_cat_nan'].sum(), 0)

        self.assertEqual(after_onehot['str_cat_a'].sum(), 2)
        self.assertEqual(after_onehot['str_cat_b'].sum(), 2)
        self.assertEqual(after_onehot['str_cat_nan'].sum(), 0)
    
    def test_can_encode_floats(self):
        """
        Ensures categorical and ordinal float values
        are properly encoded.
        """
        after_onehot = self._fit_produce(self.categorical_ordinal_float_data)

        self.assertEqual(after_onehot.shape, (4, 9))

        self.assertEqual(after_onehot['float_cat_1'].sum(), 1)
        self.assertEqual(after_onehot['float_cat_2'].sum(), 2)
        self.assertEqual(after_onehot['float_cat_3.4'].sum(), 1)
        self.assertEqual(after_onehot['float_cat_nan'].sum(), 0)

        self.assertEqual(after_onehot['float_ord_4'].sum(), 2)
        self.assertEqual(after_onehot['float_ord_5'].sum(), 1)
        self.assertEqual(after_onehot['float_ord_6.1'].sum(), 1)
        self.assertEqual(after_onehot['float_ord_nan'].sum(), 0)
    
    def test_can_encode_ordinal(self):
        """
        Ensures ordinal attributes are encoded.
        """
        after_onehot = self._fit_produce(self.ordinal_data)

        self.assertEqual(after_onehot.shape, (4, 4))

        self.assertEqual(after_onehot['ord_1'].sum(), 3)
        self.assertEqual(after_onehot['ord_2'].sum(), 1)
    
    def test_wont_encode_non_categorical(self):
        """
        Ensures only attributes that are ordinal or categorical
        are encoded.
        """
        after_onehot = self._fit_produce(self.numeric_data)
        self.assertEqual(after_onehot.shape, (4,2))
        self.assertTrue(
            self._are_lists_equal(after_onehot.columns, ['intcol', 'floatcol'])
        )
    
    def test_can_handle_unseen_values(self):
        """
        Ensures the primitive does not break when produced on data
        that has values the primitive did not see while it was
        fitting.
        """
        one_hotter = self._fit(self.categorical_data)
        after_onehot = one_hotter.produce(inputs=self.categorical_data2).value

        self.assertEqual(after_onehot.shape, (4, 7))

        self.assertEqual(after_onehot['int_cat_1'].sum(), 2)
        self.assertEqual(after_onehot['int_cat_2'].sum(), 1)
        self.assertEqual(after_onehot['int_cat_nan'].sum(), 0)

        self.assertEqual(after_onehot['str_cat_a'].sum(), 1)
        self.assertEqual(after_onehot['str_cat_b'].sum(), 2)
        self.assertEqual(after_onehot['str_cat_nan'].sum(), 0)
    
    def test_can_handle_nans(self):
        """
        Ensures that nans get their own column and are properly
        encoded.
        """
        after_onehot = self._fit_produce(self.categorical_with_nans_data)

        self.assertEqual(after_onehot.shape, (5, 4))

        self.assertEqual(after_onehot['cat_4'].sum(), 2)
        self.assertEqual(after_onehot['cat_5'].sum(), 1)
        self.assertEqual(after_onehot['cat_nan'].sum(), 2)
    
    def test_column_renaming_doesnt_hurt(self):
        """
        Ensures there are no detrimental side effects when removing
        '.0' from the end of column names.
        """
        after_onehot = self._fit_produce(self.str_edge_case_data)

        self.assertEqual(after_onehot.shape, (5, 5))

        self.assertEqual(after_onehot['str_cat_a'].sum(), 2)
        self.assertEqual(after_onehot['str_cat_a.0'].sum(), 2)
        self.assertEqual(after_onehot['str_cat_b'].sum(), 1)
        self.assertEqual(after_onehot['str_cat_nan'].sum(), 0)
    
    def _are_lists_equal(self, l1, l2) -> bool:
        if len(l1) != len(l2):
            return False
        return all(a == b for a, b in zip(l1, l2))
    
    def _fit(self, df: DataFrame):
        """Returns an encoder fitted on `df`."""
        one_hotter = Encoder(hyperparams=self.hyperparams_class.defaults())
        one_hotter.set_training_data(inputs=df)
        one_hotter.fit()
        return one_hotter
    
    def _fit_produce(self, df: DataFrame) -> DataFrame:
        """
        Fits and produces an encoder on `df`, returning the produced results.
        """
        one_hotter = self._fit(df)
        return one_hotter.produce(inputs=df).value
    
    def _make_df(self, cols: List[Column]) -> DataFrame:
        # Make the data
        df: DataFrame = DataFrame(
            pd.DataFrame({ col.name: col.data for col in cols }),
            generate_metadata=True
        )
        # Add each column's semantic types
        for col in cols:
            for semantic_type in col.semantic_types:
                df.metadata = df.metadata.add_semantic_type(
                    (metadata_base.ALL_ELEMENTS, df.columns.get_loc(col.name)),
                    semantic_type
                )
        return df

    def _print_df(self, df: DataFrame, name: str) -> None:
        """Debug helper"""
        print(f'\n"{name}" Data Frame:')
        print(df)
        print('column metadata:')
        for col_i in range(df.shape[1]):
            print(df.metadata.query_column(col_i))

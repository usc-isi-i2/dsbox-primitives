import logging
import numpy as np
_logger = logging.getLogger(__name__)


class TemplateSteps:

    '''
    Some steps and parameters that are used for creating templates
    Returns a list of dicts with the most common steps
    '''
    @staticmethod
    def dsbox_generic_steps3(data: str = "data", target: str = "target"):
        '''
        dsbox generic step for classification and regression, directly lead to model step
        '''
        return [
            {
                "name": "sampling_step",
                "primitives": ["d3m.primitives.data_preprocessing.splitter.DSBOX"],
                "inputs": ["template_input"]
            },
            {
                "name": "denormalize_step",
                "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                "inputs": ["sampling_step"]
            },
            {
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["denormalize_step"]
            },
            {
                "name": "common_profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                'https://metadata.datadrivendiscovery.org/types/Attribute',
                                ),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encode_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.label_encoder.DSBOX"
                ],
                "inputs": ["clean_step"]
            },
            # {
            #     "name": "corex_step",
            #     "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
            #     "inputs": ["encode_step"]
            # },
            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                "inputs":["encode_step"],
            },
            {
                "name": "pre_"+target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": data,
                "primitives": ["d3m.primitives.data_preprocessing.greedy_imputation.DSBOX"],
                "inputs": ["to_numeric_step", "pre_" + target]
            },
            {
                "name": target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_"+target]
            },
        ]

    @staticmethod
    def dsbox_generic_steps2(data: str = "data", target: str = "target"):
        '''
        dsbox generic step for classification and regression, directly lead to model step
        '''
        return [
            {
                "name": "sampling_step",
                "primitives": ["d3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX"],
                "inputs": ["template_input"]
            },
            {
                "name": "denormalize_step",
                "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                "inputs": ["sampling_step"]
            },
            {
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["denormalize_step"]
            },
            {
                "name": "common_profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                'https://metadata.datadrivendiscovery.org/types/Attribute',
                                ),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                    # "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encode_step",
                "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX"],
                "inputs": ["clean_step"]
            },
            # {
            #     "name": "corex_step",
            #     "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
            #     "inputs": ["encode_step"]
            # },
            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                "inputs":["encode_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.iterative_regression_imputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": "scaler_step",
                "primitives": [
                    {
                        "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                        "hyperparameters": {}
                    },
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": data,
                "primitives": [
                    # 19 Feb 2019: Stop using PCA until issue is resolved
                    # https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues/154
                    # {
                    #     "primitive": "d3m.primitives.data_transformation.pca.SKlearn",
                    #     "hyperparameters":
                    #     {
                    #         'use_semantic_types': [True],
                    #         'add_index_columns': [True],
                    #         'return_result': ['new'],
                    #         'n_components': [10, 15, 25]
                    #     }
                    # },
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["scaler_step"]
            },
            {
                "name": "pre_"+target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_"+target]
            },
        ]


    @staticmethod
    def dsbox_generic_steps(data: str = "data", target: str = "target"):
        '''
        dsbox generic step for classification and regression, directly lead to model step
        '''
        return [
            {
                "name": "sampling_step",
                "primitives": ["d3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX"],
                "inputs": ["template_input"]
            },

            {
                "name": "denormalize_step",
                "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                "inputs": ["sampling_step"]
            },
            {
                "name": "to_dataframe_step",
                "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                "inputs": ["denormalize_step"]
            },
            {
                "name": "common_profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                "inputs": ["to_dataframe_step"]
            },
            {
                "name": "extract_attribute_step",
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                'https://metadata.datadrivendiscovery.org/types/Attribute',
                                ),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": "profiler_step",
                "primitives": ["d3m.primitives.schema_discovery.profiler.DSBOX"],
                "inputs": ["extract_attribute_step"]
            },
            {
                "name": "clean_step",
                "primitives": [
                    "d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX",
                    # "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["profiler_step"]
            },
            {
                "name": "encode_step",
                "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                "inputs": ["clean_step"]
            },
            # {
            #     "name": "corex_step",
            #     "primitives": ["d3m.primitives.feature_construction.corex_text.DSBOX"],
            #     "inputs": ["encode_step"]
            # },
            {
                "name": "to_numeric_step",
                "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                "inputs":["clean_step"],
            },
            {
                "name": "impute_step",
                "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                "inputs": ["to_numeric_step"]
            },
            {
                "name": "scaler_step",
                "primitives": [
                    # {
                    #     "primitive": "d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn",
                    #     "hyperparameters":
                    #     {
                    #         'use_semantic_types':[True],
                    #         'return_result':['new'],
                    #         'add_index_columns':[True],
                    #     }
                    # },
                    {
                        "primitive": "d3m.primitives.normalization.iqr_scaler.DSBOX",
                        "hyperparameters": {}
                    },
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["impute_step"]
            },
            {
                "name": data,
                "primitives": [
                    # 19 Feb 2019: Stop using PCA until issue is resolved
                    # https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues/154
                    # {
                    #     "primitive": "d3m.primitives.data_transformation.pca.SKlearn",
                    #     "hyperparameters":
                    #     {
                    #         'use_semantic_types': [True],
                    #         'add_index_columns': [True],
                    #         'return_result': ['new'],
                    #         'n_components': [10, 15, 25]
                    #     }
                    # },
                    "d3m.primitives.data_preprocessing.do_nothing.DSBOX",
                ],
                "inputs": ["scaler_step"]
            },
            {
                "name": "pre_"+target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                    "hyperparameters":
                        {
                            'semantic_types': (
                                'https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                            'use_columns': (),
                            'exclude_columns': ()
                        }
                }],
                "inputs": ["common_profiler_step"]
            },
            {
                "name": target,
                "primitives": [{
                    "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                    "hyperparameters": {
                        "drop_non_numeric_columns": [False]
                    }
                }],
                "inputs": ["pre_"+target]
            },
        ]

    @staticmethod
    def class_hyperparameter_generator(primitive_name, parameter_name, definition):
        from d3m import index
        g = None
        try:
            g = index.get_primitive(primitive_name).metadata.query()["primitive_code"]["hyperparams"][parameter_name]['structural_type'](definition)
        except Exception:
            _logger.error(f"Hyperparameter not valid for {primitive_name}!")
            pass
        return g


    @staticmethod
    def dsbox_feature_selector(ptype, first_input='impute_step', second_input='extract_target_step'):
        '''
        dsbox feature selection steps for classification and regression, lead to feature selector steps
        '''
        if ptype == "regression":
            return [
                {
                    "name": "feature_selector_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_selection.select_fwe.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.generic_univariate_select.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'add_index_columns': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "score_func": ["f_regression"],
                                "mode": ["percentile"],
                                "param": [5, 7, 10, 15, 30, 50, 75],
                            }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.joint_mutual_information.AutoRPI",
                            "hyperparameters": {
                                #'method': ["counting", "pseudoBayesian", "fullBayesian"],
                                'nbins': [2, 5, 10, 13, 20]
                                }
                        },
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX"
                    ],
                    "inputs":[first_input, second_input]
                },
            ]
        else:
            return [
                {
                    "name": "feature_selector_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_selection.select_fwe.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "alpha": [float(x) for x in np.logspace(-4, -1, 6)]
                            }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.generic_univariate_select.SKlearn",
                            "hyperparameters": {
                                'use_semantic_types': [True],
                                'return_result': ['new'],
                                'add_index_columns': [True],
                                "mode": ["percentile"],
                                "param": [5, 7, 10, 15, 30, 50, 75],
                            }
                        },
                        {
                            "primitive": "d3m.primitives.feature_selection.joint_mutual_information.AutoRPI",
                            "hyperparameters": {
                                #'method': ["counting", "pseudoBayesian", "fullBayesian"],
                                'nbins': [2, 5, 10, 13, 20]
                                }
                        },
                        "d3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI",
                        "d3m.primitives.data_preprocessing.do_nothing.DSBOX"

                    ],
                    "inputs":[first_input, second_input]
                },
            ]
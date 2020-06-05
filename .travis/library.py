from d3m.metadata.problem import TaskKeyword
from template import DSBoxTemplate
from template_steps import TemplateSteps


class DefaultClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_classification_template",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "runType": "classification",
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": (TemplateSteps.dsbox_generic_steps()
                      + TemplateSteps.dsbox_feature_selector("classification", first_input='data', second_input='target')
                      + [
                          {
                              "name": "model_step",
                              "runtime": {
                                  "cross_validation": 10,
                                  "stratified": True
                              },
                              "primitives": [
                                  {
                                      "primitive":
                                      "d3m.primitives.classification.gradient_boosting.SKlearn",
                                      "hyperparameters":
                                      {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                          'max_depth': [5],
                                          'learning_rate':[0.1],
                                          'min_samples_leaf': [2],
                                          'min_samples_split': [3],
                                          'n_estimators': [50],
                                         }
                                  }
                              ],
                              "inputs": ["feature_selector_step", "target"]
                          }
                      ])
        }


class DefaultClassificationTemplate2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_classification_template2",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "runType": "classification",
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": (TemplateSteps.dsbox_generic_steps2()
                      + TemplateSteps.dsbox_feature_selector("classification", first_input='data', second_input='target')
                      + [
                          {
                              "name": "model_step",
                              "runtime": {
                                  "cross_validation": 10,
                                  "stratified": True
                              },
                              "primitives": [
                                  {
                                      "primitive":
                                      "d3m.primitives.classification.gradient_boosting.SKlearn",
                                      "hyperparameters":
                                      {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                          'max_depth': [5],
                                          'learning_rate':[0.1],
                                          'min_samples_leaf': [2],
                                          'min_samples_split': [3],
                                          'n_estimators': [50],
                                         }
                                  }
                              ],
                              "inputs": ["feature_selector_step", "target"]
                          }
                      ])
        }


class DefaultTimeseriesCollectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "Default_timeseries_collection_template",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "runType": "timeseries",
            "inputType": {"timeseries", "table"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "random_forest_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [

                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },
                # {
                #     "name": "column_parser_step",
                #     "primitives": ["d3m.primitives.data_transformation.column_parser.DataFrameCommon"],
                #     "inputs": ["extract_target_step"]
                # },

                # read X value
                {
                    "name": "timeseries_to_list_step",
                    "primitives": ["d3m.primitives.data_preprocessing.time_series_to_list.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },

                {
                    "name": "random_projection_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX",
                            "hyperparameters":{
                                'generate_metadata':[True],
                            }
                        }
                    ],
                    "inputs": ["timeseries_to_list_step"]
                },

                {
                    "name": "random_forest_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types':[True],
                            }
                        }
                    ],
                    "inputs": ["random_projection_step", "extract_target_step"]
                },
            ]
        }


class DefaultRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_regression_template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "runType": "regression",
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "test_validation": 5,
                        "stratified": False
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 4, 5],
                                    'n_estimators': [100, 130, 165, 200],
                                    'learning_rate': [0.1, 0.23, 0.34, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class DefaultRegressionTemplate2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_regression_template2",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "runType": "regression",
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": TemplateSteps.dsbox_generic_steps3() + [
                {
                    "name": "model_step",
                    "runtime": {
                        "test_validation": 5,
                        "stratified": False
                    },
                    "primitives": [
                        {
                            "primitive":
                                "d3m.primitives.regression.gradient_boosting.SKlearn",
                            "hyperparameters":
                                {
                                    'max_depth': [2, 3, 4, 5],
                                    'n_estimators': [100, 130, 165, 200],
                                    'learning_rate': [0.1, 0.23, 0.34, 0.5],
                                    'min_samples_split': [2, 3],
                                    'min_samples_leaf': [1, 2],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.extra_trees.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                        {
                            "primitive":
                                "d3m.primitives.regression.random_forest.SKlearn",
                            "hyperparameters":
                                {
                                    'bootstrap': [True, False],
                                    'max_depth': [15, 30, None],
                                    'min_samples_leaf': [1, 2, 4],
                                    'min_samples_split': [2, 5, 10],
                                    'max_features': ['auto', 'sqrt'],
                                    'n_estimators': [10, 50, 100],
                                    'add_index_columns': [True],
                                    'use_semantic_types':[True],
                                }
                        },
                    ],
                    "inputs": ["data", "target"]
                }
            ]
        }


class VotingTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "default_classification_template",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "runType": "voting_classification",
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": (TemplateSteps.dsbox_generic_steps()
                      + TemplateSteps.dsbox_feature_selector("classification", first_input='data', second_input='target')
                      + [
                          {
                              "name": "model_substep_1",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.linear_discriminant_analysis.SKlearn",
                                      "hyperparameters": {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                      }
                                  }],
                              "inputs": ["feature_selector_step", "target"]
                          },
                          {
                              "name": "model_substep_2",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.nearest_centroid.SKlearn",
                                      "hyperparameters": {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                      }
                                  }],
                              "inputs": ["feature_selector_step", "target"]
                          },
                          {
                              "name": "model_substep_3",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.logistic_regression.SKlearn",
                                      "hyperparameters": {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                      }
                                  }],
                              "inputs": ["feature_selector_step", "target"]
                          },
                          {
                              "name": "vertical_concat",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.data_preprocessing.vertical_concatenate.DSBOX",
                                      "hyperparameters": {}
                                  }],
                              "inputs": [["model_substep_1", "model_substep_2", "model_substep_3"]]
                          },
                          {
                              "name": "model_step",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.ensemble_voting.DSBOX",
                                      "hyperparameters": {}
                                  }],
                              "inputs": ["vertical_concat", "target"]
                          }
                      ])
        }

class HorizontalVotingTemplate(DSBoxTemplate):
    'Horizontal Voting Template'
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "horizontal_voting_template",
            "taskSubtype": {TaskKeyword.BINARY.name, TaskKeyword.MULTICLASS.name},
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "runType": "voting_classification",
            "inputType": "table",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "target",  # Name of the step generating the ground truth
            "steps": (TemplateSteps.dsbox_generic_steps()
                      + TemplateSteps.dsbox_feature_selector("classification", first_input='data', second_input='target')
                      + [
                          {
                              "name": "model_substep_1",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.linear_discriminant_analysis.SKlearn",
                                      "hyperparameters": {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                      }
                                  }],
                              "inputs": ["feature_selector_step", "target"]
                          },
                          {
                              "name": "model_substep_2",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.nearest_centroid.SKlearn",
                                      "hyperparameters": {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                      }
                                  }],
                              "inputs": ["feature_selector_step", "target"]
                          },
                          {
                              "name": "model_substep_3",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.logistic_regression.SKlearn",
                                      "hyperparameters": {
                                          'use_semantic_types': [True],
                                          'return_result': ['new'],
                                          'add_index_columns': [True],
                                      }
                                  }],
                              "inputs": ["feature_selector_step", "target"]
                          },
                          {
                              "name": "horizontal_concat",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.data_preprocessing.horizontal_concat.DSBOX",
                                      "hyperparameters": {}
                                  }],
                              "inputs": [["feature_selector_step", "model_substep_1", "model_substep_2", "model_substep_3"]]
                          },
                          {
                              "name": "to_attribute",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.data_transformation.replace_semantic_types.Common",
                                      "hyperparameters": {
                                          'from_semantic_types': [[
                                              "https://metadata.datadrivendiscovery.org/types/Target",
                                              "https://metadata.datadrivendiscovery.org/types/PredictedTarget"
                                          ]],
                                          'to_semantic_types': [[
                                              "https://metadata.datadrivendiscovery.org/types/Attribute"
                                          ]],
                                          'match_logic': ['any'],
                                      }
                                  }
                              ],
                              "inputs": ["horizontal_concat"]
                          },
                          {
                              "name": "encode_new_columns",
                              "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                              "inputs": ["to_attribute"]
                          },
                          {
                              "name": "model_step",
                              "primitives": [
                                  {
                                      "primitive": "d3m.primitives.classification.random_forest.SKlearn",
                                      "hyperparameters": {
                                          'add_index_columns': [True],
                                          'use_semantic_types':[True],
                                      }
                                  }
                              ],
                              "inputs": ["encode_new_columns", "target"]
                          }
                      ])
        }

class TA1ImageProcessingRegressionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1ImageProcessingRegressionTemplate",
            "taskType": TaskKeyword.REGRESSION.name,
            "runType": "image_regression",
            # See TaskKeyword, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "regressor_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.resnet50_image_feature.DSBOX",
                            "hyperparameters": {
                                'generate_metadata': [True]
                            }
                        }
                    ],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types': [True]
                            }
                        }
                    ],
                    "inputs": ["feature_extraction"]
                },
                {
                    "name": "regressor_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.ridge.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                            }
                        }
                    ],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }


class TA1ImageProcessingRegressionTemplate2(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "TA1ImageProcessingRegressionTemplate2",
            "taskType": TaskKeyword.REGRESSION.name,
            "runType": "image_regression",
            # See TaskKeyword, range include 'CLASSIFICATION', 'CLUSTERING', 'COLLABORATIVE_FILTERING', 'COMMUNITY_DETECTION', 'GRAPH_CLUSTERING', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'REGRESSION', 'TIME_SERIES', 'VERTEX_NOMINATION'
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "inputType": "image",  # See SEMANTIC_TYPES.keys() for range of values
            "output": "regressor_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read Y value
                {
                    "name": "pre_extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                             }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.to_numeric.DSBOX",
                        "hyperparameters": {
                            "drop_non_numeric_columns": [False]
                        }
                    }],
                    "inputs": ["pre_extract_target_step"]
                },
                {
                    "name": "dataframe_to_tensor",
                    "primitives": ["d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "feature_extraction",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.vgg16_image_feature.DSBOX",
                            "hyperparameters": {
                                'generate_metadata': [True]
                            }
                        }
                    ],
                    "inputs": ["dataframe_to_tensor"]
                },
                {
                    "name": "PCA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.pca.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                                'use_semantic_types': [True]
                            }
                        }
                    ],
                    "inputs": ["feature_extraction"]
                },
                {
                    "name": "regressor_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.regression.ridge.SKlearn",
                            "hyperparameters": {
                                'add_index_columns': [True],
                            }
                        }
                    ],
                    "inputs": ["PCA_step", "extract_target_step"]
                },
            ]
        }


class ARIMATemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "ARIMA_Template",
            "taskType": TaskKeyword.TIME_SERIES.name,
            "runType": "time_series_forecasting",
            "taskSubtype": "NONE",
            "inputType": {"table", "timeseries"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "ARIMA_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "parser_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.column_parser.Common",
                        "hyperparameters": {
                            "parse_semantic_types": [('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector'),]
                        }
                    }],
                    "inputs": ["to_dataframe_step"]
                },

                # read Y value
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                             'use_columns': (),
                             'exclude_columns': ()
                            }
                    }],
                    "inputs": ["parser_step"]
                },

                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                                   'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                                                   'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["parser_step"]
                },
                # {
                #     "name": "extract_attribute_step",
                #     "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                #     "inputs": ["pre_extract_attribute_step"]
                # },
                {
                    "name": "ARIMA_step",
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.time_series_forecasting.arima.DSBOX",
                            "hyperparameters": {
                                "take_log": [(False)],
                                "auto": [(True)]
                            }
                        }
                    ], # can add tuning parameters like: auto-fit, take_log, etc
                    "inputs": ["extract_attribute_step", "extract_target_step"]
                },
            ]
        }


class DefaultObjectDetectionTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultObjectDetectionTemplate",
            "taskType": TaskKeyword.OBJECT_DETECTION.name,
            "runType": "object_detection",
            "taskSubtype": "NONE",
            "inputType": {"table", "image"},  # See SEMANTIC_TYPES.keys() for range of values
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",#step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read X value
                {
                    "name": "extract_file_step",#step 2
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",# step 3
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
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "model_step", # step 4
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.feature_extraction.yolo.DSBOX",
                            "hyperparameters": {
                            "use_fitted_weight": [(True)],
                            # "confidences_threshold": [(0.8)],
                            # "nms_threshold": [0.6],
                            }
                        }
                    ],
                    "inputs": ["extract_file_step", "extract_target_step"]
                },
            ]
        }


class DefaultVideoClassificationTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "DefaultVideoClassificationTemplate",
            "taskType": TaskKeyword.CLASSIFICATION.name,
            "taskSubtype": TaskKeyword.MULTICLASS.name,
            "runType": "video_classification",
            "inputType": "video",
            "output": "model_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "to_dataframe_step",#step 1
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["denormalize_step"]
                },
                # read X value
                {
                    "name": "extract_file_step",#step 2
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/FileName',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "extract_target_step",# step 3
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
                    "inputs": ["to_dataframe_step"]
                },
                {
                    "name": "video_reader",#step 4
                    "primitives": ["d3m.primitives.data_preprocessing.video_reader.Common"],
                    "inputs": ["extract_file_step"]
                },
                {
                    "name": "video_feature_extract",#step 5
                    "primitives": [
                            {
                                "primitive": "d3m.primitives.feature_extraction.inceptionV3_image_feature.DSBOX",
                                "hyperparameters": {
                                    "use_limitation":[(True),],
                                }
                            }

                        ],
                    "inputs": ["video_reader"]
                },
                {
                    "name": "model_step", # step 6
                    "primitives": [
                        {
                            "primitive": "d3m.primitives.classification.lstm.DSBOX",
                            "hyperparameters": {
                                "LSTM_units":[2048],
                                "epochs":[50, 500, 1000],
                            }
                        }
                    ],
                    "inputs": ["video_feature_extract", "extract_target_step"]
                },
            ]
        }


class UU3TestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "UU3_Test_Template",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",
            "runType": "multitable_dataset",
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                # {
                #     "name": "denormalize_step",
                #     "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                #     "inputs": ["template_input"]
                # },
                {
                    "name": "multi_table_processing_step",
                    "primitives": ["d3m.primitives.feature_extraction.multitable_featurization.DSBOX"],
                    "inputs": ["template_input"]
                },
                # {
                #     "name": "to_dataframe_step",
                #     "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                #     "inputs": ["multi_table_processing_step"]
                # },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["multi_table_processing_step"]
                },
                {
                    "name": "encode1_step",
                    "primitives": ["d3m.primitives.data_preprocessing.unary_encoder.DSBOX"],
                    "inputs": ["extract_attribute_step"]
                },
                {
                    "name": "encode2_step",
                    "primitives": ["d3m.primitives.data_preprocessing.encoder.DSBOX"],
                    "inputs": ["encode1_step"]
                },
                {
                    "name": "to_numeric_step",
                    "primitives": ["d3m.primitives.data_transformation.to_numeric.DSBOX"],
                    "inputs":["encode2_step"],
                },
                {
                    "name": "impute_step",
                    "primitives": ["d3m.primitives.data_preprocessing.mean_imputation.DSBOX"],
                    "inputs": ["to_numeric_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                        "hyperparameters":
                            {
                              'use_semantic_types': [True],
                              'return_result': ['new'],
                              'add_index_columns': [True],
                            }
                    }
                    ],
                    "inputs": ["impute_step", "extract_target_step"]
                },
                {
                  "name": "construct_prediction_step",
                    "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                    "inputs": ["model_step", "common_profiler_step"]
                }
            ]
        }


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

class AugmentTestTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "AugmentTestTemplate",
            "taskSubtype": {TaskKeyword.UNIVARIATE.name, TaskKeyword.MULTIVARIATE.name},
            "taskType": TaskKeyword.REGRESSION.name,
            "inputType": "table",
            "runType": "augment_dataset",
            "output": "construct_prediction_step",  # Name of the final step generating the prediction
            "target": "extract_target_step",  # Name of the step generating the ground truth
            "steps": [
                {
                    "name": "denormalize_step",
                    "primitives": ["d3m.primitives.data_transformation.denormalize.Common"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "augment_step2",
                    "primitives": [{"primitive": "d3m.primitives.data_augmentation.datamart_augmentation.Common", "hyperparameters": {"system_identifier": ["ISI"], "search_result": ["{\"all_column_names\": {\"left_names\": [\"d3mIndex\", \"UNITID\", \"INSTNM\", \"PCTFLOAN\", \"CONTROL\", \"STABBR\", \"PCIP16\", \"MD_EARN_WNE_P10\", \"PPTUG_EF\", \"UGDS_WHITE\", \"UGDS_BLACK\", \"UGDS_HISP\", \"UGDS_ASIAN\", \"SATMTMID\", \"SATVRMID\", \"SATWRMID\", \"UGDS\", \"PREDDEG\", \"DEBT_EARNINGS_RATIO\", \"STABBR_wikidata\"], \"right_names\": [\"UNITID\", \"OPEID\", \"OPEID6\", \"INSTNM\", \"CITY\", \"STABBR\", \"INSTURL\", \"NPCURL\", \"HCM2\", \"PREDDEG\", \"HIGHDEG\", \"CONTROL\", \"LOCALE\", \"HBCU\", \"PBI\", \"ANNHI\", \"TRIBAL\", \"AANAPII\", \"HSI\", \"NANTI\", \"MENONLY\", \"WOMENONLY\", \"RELAFFIL\", \"SATVR25\", \"SATVR75\", \"SATMT25\", \"SATMT75\", \"SATWR25\", \"SATWR75\", \"SATVRMID\", \"SATMTMID\", \"SATWRMID\", \"ACTCM25\", \"ACTCM75\", \"ACTEN25\", \"ACTEN75\", \"ACTMT25\", \"ACTMT75\", \"ACTWR25\", \"ACTWR75\", \"ACTCMMID\", \"ACTENMID\", \"ACTMTMID\", \"ACTWRMID\", \"SAT_AVG\", \"SAT_AVG_ALL\", \"PCIP01\", \"PCIP03\", \"PCIP04\", \"PCIP05\", \"PCIP09\", \"PCIP10\", \"PCIP11\", \"PCIP12\", \"PCIP13\", \"PCIP14\", \"PCIP15\", \"PCIP16\", \"PCIP19\", \"PCIP22\", \"PCIP23\", \"PCIP24\", \"PCIP25\", \"PCIP26\", \"PCIP27\", \"PCIP29\", \"PCIP30\", \"PCIP31\", \"PCIP38\", \"PCIP39\", \"PCIP40\", \"PCIP41\", \"PCIP42\", \"PCIP43\", \"PCIP44\", \"PCIP45\", \"PCIP46\", \"PCIP47\", \"PCIP48\", \"PCIP49\", \"PCIP50\", \"PCIP51\", \"PCIP52\", \"PCIP54\", \"DISTANCEONLY\", \"UGDS\", \"UGDS_WHITE\", \"UGDS_BLACK\", \"UGDS_HISP\", \"UGDS_ASIAN\", \"UGDS_AIAN\", \"UGDS_NHPI\", \"UGDS_2MOR\", \"UGDS_NRA\", \"UGDS_UNKN\", \"PPTUG_EF\", \"CURROPER\", \"NPT4_PUB\", \"NPT4_PRIV\", \"NPT41_PUB\", \"NPT42_PUB\", \"NPT43_PUB\", \"NPT44_PUB\", \"NPT45_PUB\", \"NPT41_PRIV\", \"NPT42_PRIV\", \"NPT43_PRIV\", \"NPT44_PRIV\", \"NPT45_PRIV\", \"PCTPELL\", \"RET_FT4_POOLED_SUPP\", \"RET_FTL4_POOLED_SUPP\", \"RET_PT4_POOLED_SUPP\", \"RET_PTL4_POOLED_SUPP\", \"PCTFLOAN\", \"UG25ABV\", \"MD_EARN_WNE_P10\", \"GT_25K_P6\", \"GT_28K_P6\", \"GRAD_DEBT_MDN_SUPP\", \"GRAD_DEBT_MDN10YR_SUPP\", \"RPY_3YR_RT_SUPP\", \"C150_L4_POOLED_SUPP\", \"C150_4_POOLED_SUPP\", \"UNITID_wikidata\", \"OPEID6_wikidata\", \"STABBR_wikidata\", \"CITY_wikidata\"]}, \"augmentation\": {\"left_columns\": [[2]], \"right_columns\": [[3]], \"type\": \"join\"}, \"file_type\": \"csv\", \"id\": \"D4cb70062-77ed-4097-a486-0b43ffe81463\", \"materialize_info\": \"{\\\"id\\\": \\\"D4cb70062-77ed-4097-a486-0b43ffe81463\\\", \\\"score\\\": 0.9050571841886983, \\\"metadata\\\": {\\\"connection_url\\\": \\\"https://dsbox02.isi.edu:9000\\\", \\\"search_result\\\": {\\\"variable\\\": {\\\"type\\\": \\\"uri\\\", \\\"value\\\": \\\"http://www.wikidata.org/entity/statement/D4cb70062-77ed-4097-a486-0b43ffe81463-db0080de-12d9-4189-b13a-2a46fa63a227\\\"}, \\\"dataset\\\": {\\\"type\\\": \\\"uri\\\", \\\"value\\\": \\\"http://www.wikidata.org/entity/D4cb70062-77ed-4097-a486-0b43ffe81463\\\"}, \\\"url\\\": {\\\"type\\\": \\\"uri\\\", \\\"value\\\": \\\"http://dsbox02.isi.edu:9000/upload/local_datasets/Most-Recent-Cohorts-Scorecard-Elements.csv\\\"}, \\\"file_type\\\": {\\\"datatype\\\": \\\"http://www.w3.org/2001/XMLSchema#string\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"csv\\\"}, \\\"extra_information\\\": {\\\"datatype\\\": \\\"http://www.w3.org/2001/XMLSchema#string\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"{\\\\\\\"column_meta_0\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UNITID\\\\\\\"}, \\\\\\\"column_meta_1\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"OPEID\\\\\\\"}, \\\\\\\"column_meta_2\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"OPEID6\\\\\\\"}, \\\\\\\"column_meta_3\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"INSTNM\\\\\\\"}, \\\\\\\"column_meta_4\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"CITY\\\\\\\"}, \\\\\\\"column_meta_5\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"STABBR\\\\\\\"}, \\\\\\\"column_meta_6\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"INSTURL\\\\\\\"}, \\\\\\\"column_meta_7\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPCURL\\\\\\\"}, \\\\\\\"column_meta_8\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"HCM2\\\\\\\"}, \\\\\\\"column_meta_9\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PREDDEG\\\\\\\"}, \\\\\\\"column_meta_10\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"HIGHDEG\\\\\\\"}, \\\\\\\"column_meta_11\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"CONTROL\\\\\\\"}, \\\\\\\"column_meta_12\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"LOCALE\\\\\\\"}, \\\\\\\"column_meta_13\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"HBCU\\\\\\\"}, \\\\\\\"column_meta_14\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PBI\\\\\\\"}, \\\\\\\"column_meta_15\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ANNHI\\\\\\\"}, \\\\\\\"column_meta_16\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"TRIBAL\\\\\\\"}, \\\\\\\"column_meta_17\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"AANAPII\\\\\\\"}, \\\\\\\"column_meta_18\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"HSI\\\\\\\"}, \\\\\\\"column_meta_19\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NANTI\\\\\\\"}, \\\\\\\"column_meta_20\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"MENONLY\\\\\\\"}, \\\\\\\"column_meta_21\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"WOMENONLY\\\\\\\"}, \\\\\\\"column_meta_22\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"RELAFFIL\\\\\\\"}, \\\\\\\"column_meta_23\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATVR25\\\\\\\"}, \\\\\\\"column_meta_24\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATVR75\\\\\\\"}, \\\\\\\"column_meta_25\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATMT25\\\\\\\"}, \\\\\\\"column_meta_26\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATMT75\\\\\\\"}, \\\\\\\"column_meta_27\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATWR25\\\\\\\"}, \\\\\\\"column_meta_28\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATWR75\\\\\\\"}, \\\\\\\"column_meta_29\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATVRMID\\\\\\\"}, \\\\\\\"column_meta_30\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATMTMID\\\\\\\"}, \\\\\\\"column_meta_31\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SATWRMID\\\\\\\"}, \\\\\\\"column_meta_32\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTCM25\\\\\\\"}, \\\\\\\"column_meta_33\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTCM75\\\\\\\"}, \\\\\\\"column_meta_34\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTEN25\\\\\\\"}, \\\\\\\"column_meta_35\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTEN75\\\\\\\"}, \\\\\\\"column_meta_36\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTMT25\\\\\\\"}, \\\\\\\"column_meta_37\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTMT75\\\\\\\"}, \\\\\\\"column_meta_38\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"https://metadata.datadrivendiscovery.org/types/CategoricalData\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTWR25\\\\\\\"}, \\\\\\\"column_meta_39\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"https://metadata.datadrivendiscovery.org/types/CategoricalData\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTWR75\\\\\\\"}, \\\\\\\"column_meta_40\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTCMMID\\\\\\\"}, \\\\\\\"column_meta_41\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTENMID\\\\\\\"}, \\\\\\\"column_meta_42\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTMTMID\\\\\\\"}, \\\\\\\"column_meta_43\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"https://metadata.datadrivendiscovery.org/types/CategoricalData\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"ACTWRMID\\\\\\\"}, \\\\\\\"column_meta_44\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SAT_AVG\\\\\\\"}, \\\\\\\"column_meta_45\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"SAT_AVG_ALL\\\\\\\"}, \\\\\\\"column_meta_46\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP01\\\\\\\"}, \\\\\\\"column_meta_47\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP03\\\\\\\"}, \\\\\\\"column_meta_48\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP04\\\\\\\"}, \\\\\\\"column_meta_49\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP05\\\\\\\"}, \\\\\\\"column_meta_50\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP09\\\\\\\"}, \\\\\\\"column_meta_51\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP10\\\\\\\"}, \\\\\\\"column_meta_52\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP11\\\\\\\"}, \\\\\\\"column_meta_53\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP12\\\\\\\"}, \\\\\\\"column_meta_54\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP13\\\\\\\"}, \\\\\\\"column_meta_55\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP14\\\\\\\"}, \\\\\\\"column_meta_56\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP15\\\\\\\"}, \\\\\\\"column_meta_57\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP16\\\\\\\"}, \\\\\\\"column_meta_58\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP19\\\\\\\"}, \\\\\\\"column_meta_59\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP22\\\\\\\"}, \\\\\\\"column_meta_60\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP23\\\\\\\"}, \\\\\\\"column_meta_61\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP24\\\\\\\"}, \\\\\\\"column_meta_62\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP25\\\\\\\"}, \\\\\\\"column_meta_63\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP26\\\\\\\"}, \\\\\\\"column_meta_64\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP27\\\\\\\"}, \\\\\\\"column_meta_65\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP29\\\\\\\"}, \\\\\\\"column_meta_66\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP30\\\\\\\"}, \\\\\\\"column_meta_67\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP31\\\\\\\"}, \\\\\\\"column_meta_68\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP38\\\\\\\"}, \\\\\\\"column_meta_69\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP39\\\\\\\"}, \\\\\\\"column_meta_70\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP40\\\\\\\"}, \\\\\\\"column_meta_71\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP41\\\\\\\"}, \\\\\\\"column_meta_72\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP42\\\\\\\"}, \\\\\\\"column_meta_73\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP43\\\\\\\"}, \\\\\\\"column_meta_74\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP44\\\\\\\"}, \\\\\\\"column_meta_75\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP45\\\\\\\"}, \\\\\\\"column_meta_76\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP46\\\\\\\"}, \\\\\\\"column_meta_77\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP47\\\\\\\"}, \\\\\\\"column_meta_78\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP48\\\\\\\"}, \\\\\\\"column_meta_79\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP49\\\\\\\"}, \\\\\\\"column_meta_80\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP50\\\\\\\"}, \\\\\\\"column_meta_81\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP51\\\\\\\"}, \\\\\\\"column_meta_82\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP52\\\\\\\"}, \\\\\\\"column_meta_83\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCIP54\\\\\\\"}, \\\\\\\"column_meta_84\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"DISTANCEONLY\\\\\\\"}, \\\\\\\"column_meta_85\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS\\\\\\\"}, \\\\\\\"column_meta_86\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_WHITE\\\\\\\"}, \\\\\\\"column_meta_87\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_BLACK\\\\\\\"}, \\\\\\\"column_meta_88\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_HISP\\\\\\\"}, \\\\\\\"column_meta_89\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_ASIAN\\\\\\\"}, \\\\\\\"column_meta_90\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_AIAN\\\\\\\"}, \\\\\\\"column_meta_91\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_NHPI\\\\\\\"}, \\\\\\\"column_meta_92\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_2MOR\\\\\\\"}, \\\\\\\"column_meta_93\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_NRA\\\\\\\"}, \\\\\\\"column_meta_94\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UGDS_UNKN\\\\\\\"}, \\\\\\\"column_meta_95\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PPTUG_EF\\\\\\\"}, \\\\\\\"column_meta_96\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Integer\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"CURROPER\\\\\\\"}, \\\\\\\"column_meta_97\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT4_PUB\\\\\\\"}, \\\\\\\"column_meta_98\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT4_PRIV\\\\\\\"}, \\\\\\\"column_meta_99\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT41_PUB\\\\\\\"}, \\\\\\\"column_meta_100\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT42_PUB\\\\\\\"}, \\\\\\\"column_meta_101\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT43_PUB\\\\\\\"}, \\\\\\\"column_meta_102\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT44_PUB\\\\\\\"}, \\\\\\\"column_meta_103\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT45_PUB\\\\\\\"}, \\\\\\\"column_meta_104\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT41_PRIV\\\\\\\"}, \\\\\\\"column_meta_105\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT42_PRIV\\\\\\\"}, \\\\\\\"column_meta_106\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT43_PRIV\\\\\\\"}, \\\\\\\"column_meta_107\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT44_PRIV\\\\\\\"}, \\\\\\\"column_meta_108\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"NPT45_PRIV\\\\\\\"}, \\\\\\\"column_meta_109\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCTPELL\\\\\\\"}, \\\\\\\"column_meta_110\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"RET_FT4_POOLED_SUPP\\\\\\\"}, \\\\\\\"column_meta_111\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"RET_FTL4_POOLED_SUPP\\\\\\\"}, \\\\\\\"column_meta_112\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"RET_PT4_POOLED_SUPP\\\\\\\"}, \\\\\\\"column_meta_113\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"RET_PTL4_POOLED_SUPP\\\\\\\"}, \\\\\\\"column_meta_114\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"PCTFLOAN\\\\\\\"}, \\\\\\\"column_meta_115\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UG25ABV\\\\\\\"}, \\\\\\\"column_meta_116\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"MD_EARN_WNE_P10\\\\\\\"}, \\\\\\\"column_meta_117\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"GT_25K_P6\\\\\\\"}, \\\\\\\"column_meta_118\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"GT_28K_P6\\\\\\\"}, \\\\\\\"column_meta_119\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"GRAD_DEBT_MDN_SUPP\\\\\\\"}, \\\\\\\"column_meta_120\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"GRAD_DEBT_MDN10YR_SUPP\\\\\\\"}, \\\\\\\"column_meta_121\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"RPY_3YR_RT_SUPP\\\\\\\"}, \\\\\\\"column_meta_122\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"C150_L4_POOLED_SUPP\\\\\\\"}, \\\\\\\"column_meta_123\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Float\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"C150_4_POOLED_SUPP\\\\\\\"}, \\\\\\\"column_meta_124\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"UNITID_wikidata\\\\\\\"}, \\\\\\\"column_meta_125\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"OPEID6_wikidata\\\\\\\"}, \\\\\\\"column_meta_126\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"STABBR_wikidata\\\\\\\"}, \\\\\\\"column_meta_127\\\\\\\": {\\\\\\\"semantic_type\\\\\\\": [\\\\\\\"http://schema.org/Text\\\\\\\", \\\\\\\"https://metadata.datadrivendiscovery.org/types/Attribute\\\\\\\"], \\\\\\\"name\\\\\\\": \\\\\\\"CITY_wikidata\\\\\\\"}, \\\\\\\"data_metadata\\\\\\\": {\\\\\\\"shape_0\\\\\\\": 7175, \\\\\\\"shape_1\\\\\\\": 128}, \\\\\\\"first_10_rows\\\\\\\": \\\\\\\",UNITID,OPEID,OPEID6,INSTNM,CITY,STABBR,INSTURL,NPCURL,HCM2,PREDDEG,HIGHDEG,CONTROL,LOCALE,HBCU,PBI,ANNHI,TRIBAL,AANAPII,HSI,NANTI,MENONLY,WOMENONLY,RELAFFIL,SATVR25,SATVR75,SATMT25,SATMT75,SATWR25,SATWR75,SATVRMID,SATMTMID,SATWRMID,ACTCM25,ACTCM75,ACTEN25,ACTEN75,ACTMT25,ACTMT75,ACTWR25,ACTWR75,ACTCMMID,ACTENMID,ACTMTMID,ACTWRMID,SAT_AVG,SAT_AVG_ALL,PCIP01,PCIP03,PCIP04,PCIP05,PCIP09,PCIP10,PCIP11,PCIP12,PCIP13,PCIP14,PCIP15,PCIP16,PCIP19,PCIP22,PCIP23,PCIP24,PCIP25,PCIP26,PCIP27,PCIP29,PCIP30,PCIP31,PCIP38,PCIP39,PCIP40,PCIP41,PCIP42,PCIP43,PCIP44,PCIP45,PCIP46,PCIP47,PCIP48,PCIP49,PCIP50,PCIP51,PCIP52,PCIP54,DISTANCEONLY,UGDS,UGDS_WHITE,UGDS_BLACK,UGDS_HISP,UGDS_ASIAN,UGDS_AIAN,UGDS_NHPI,UGDS_2MOR,UGDS_NRA,UGDS_UNKN,PPTUG_EF,CURROPER,NPT4_PUB,NPT4_PRIV,NPT41_PUB,NPT42_PUB,NPT43_PUB,NPT44_PUB,NPT45_PUB,NPT41_PRIV,NPT42_PRIV,NPT43_PRIV,NPT44_PRIV,NPT45_PRIV,PCTPELL,RET_FT4_POOLED_SUPP,RET_FTL4_POOLED_SUPP,RET_PT4_POOLED_SUPP,RET_PTL4_POOLED_SUPP,PCTFLOAN,UG25ABV,MD_EARN_WNE_P10,GT_25K_P6,GT_28K_P6,GRAD_DEBT_MDN_SUPP,GRAD_DEBT_MDN10YR_SUPP,RPY_3YR_RT_SUPP,C150_L4_POOLED_SUPP,C150_4_POOLED_SUPP,UNITID_wikidata,OPEID6_wikidata,STABBR_wikidata,CITY_wikidata\\\\\\\\n0,100654,100200,1002,Alabama A & M University,Normal,AL,www.aamu.edu/,www2.aamu.edu/scripts/netpricecalc/npcalc.htm,0,3,4,1,12.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,380.0,470.0,370.0,470.0,370.0,457.0,425.0,420.0,414.0,16.0,19.0,14.0,20.0,15.0,18.0,,,18.0,17.0,17.0,,849.0,849.0,0.0448,0.0142,0.0071,0.0,0.0,0.0354,0.0401,0.0,0.1132,0.0896,0.0472,0.0,0.033,0.0,0.0094,0.066,0.0,0.0708,0.0024,0.0,0.0,0.0,0.0,0.0,0.0307,0.0,0.0472,0.0519,0.0377,0.0448,0.0,0.0,0.0,0.0,0.0283,0.0,0.1863,0.0,0.0,4616.0,0.0256,0.9129,0.0076,0.0019,0.0024,0.0017,0.0401,0.0065,0.0013,0.0877,1,15567.0,,15043.0,15491.0,17335.0,19562.0,18865.0,,,,,,0.7039,0.5774,,0.309,,0.7667,0.0859,31000,0.453,0.431,32750,348.16551225731,0.2531554273,,0.2913,Q39624632,Q17203888,Q173,Q575407\\\\\\\\n1,100663,105200,1052,University of Alabama at Birmingham,Birmingham,AL,www.uab.edu,uab.studentaidcalculator.com/survey.aspx,0,3,4,1,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,480.0,640.0,490.0,660.0,,,560.0,575.0,,21.0,28.0,22.0,30.0,19.0,26.0,,,25.0,26.0,23.0,,1125.0,1125.0,0.0,0.0,0.0,0.0005,0.036000000000000004,0.0,0.0131,0.0,0.0748,0.0599,0.0,0.0059,0.0,0.0,0.0158,0.0135,0.0,0.0734,0.009000000000000001,0.0,0.0,0.0,0.005,0.0,0.0212,0.0,0.0766,0.0243,0.0221,0.0365,0.0,0.0,0.0,0.0,0.0392,0.25,0.2072,0.0162,0.0,12047.0,0.5786,0.2626,0.0309,0.0598,0.0028,0.0004,0.0387,0.0179,0.0083,0.2578,1,16475.0,,13849.0,15385.0,18022.0,18705.0,19319.0,,,,,,0.3525,0.8007,,0.5178,,0.5179,0.2363,41200,0.669,0.631,21833,232.106797835537,0.513963161,,0.5384,Q39624677,Q17204336,Q173,Q79867\\\\\\\\n2,100690,2503400,25034,Amridge University,Montgomery,AL,www.amridgeuniversity.edu,www2.amridgeuniversity.edu:9091/,0,3,4,2,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,74.0,,,,,,,,,,,,,,,,,,,,,,,,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0889,0.0,0.0,0.0889,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4,0.0,0.0,0.0,0.0667,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3556,0.0,1.0,293.0,0.157,0.2355,0.0068,0.0,0.0,0.0034,0.0,0.0,0.5973,0.5392,1,,10155.0,,,,,,10155.0,,,,,0.6971,PrivacySuppressed,,PrivacySuppressed,,0.8436,0.8571,39600,0.658,0.542,22890,243.343773299842,0.2307692308,,PrivacySuppressed,Q39624831,Q17337864,Q173,Q29364\\\\\\\\n3,100706,105500,1055,University of Alabama in Huntsville,Huntsville,AL,www.uah.edu,finaid.uah.edu/,0,3,4,1,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,520.0,660.0,540.0,680.0,,,590.0,610.0,,25.0,31.0,24.0,33.0,23.0,29.0,,,28.0,29.0,26.0,,1257.0,1257.0,0.0,0.0,0.0,0.0,0.0301,0.0,0.0499,0.0,0.0282,0.2702,0.0,0.0151,0.0,0.0,0.0122,0.0,0.0,0.0603,0.0132,0.0,0.0,0.0,0.0113,0.0,0.0226,0.0,0.016,0.0,0.0,0.0188,0.0,0.0,0.0,0.0,0.0264,0.1911,0.225,0.0094,0.0,6346.0,0.7148,0.1131,0.0411,0.0414,0.012,0.0,0.0181,0.0303,0.0292,0.1746,1,19423.0,,15971.0,18016.0,20300.0,21834.0,22059.0,,,,,,0.2949,0.8161,,0.5116,,0.4312,0.2255,46700,0.685,0.649,22647,240.760438353933,0.5485090298,,0.4905,Q39624901,Q17204354,Q173,Q79860\\\\\\\\n4,100724,100500,1005,Alabama State University,Montgomery,AL,www.alasu.edu,www.alasu.edu/cost-aid/forms/calculator/index.aspx,0,3,4,1,12.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,370.0,460.0,360.0,460.0,,,415.0,410.0,,15.0,19.0,14.0,19.0,15.0,17.0,,,17.0,17.0,16.0,,825.0,825.0,0.0,0.0,0.0,0.0,0.1023,0.0,0.0503,0.0,0.1364,0.0,0.0,0.0,0.0,0.0,0.0114,0.0,0.0,0.0779,0.0146,0.0,0.0,0.0211,0.0,0.0,0.0244,0.0,0.0503,0.1412,0.0633,0.013000000000000001,0.0,0.0,0.0,0.0,0.0487,0.1429,0.0974,0.0049,0.0,4704.0,0.0138,0.9337,0.0111,0.0028,0.0013,0.0004,0.0111,0.0159,0.01,0.0727,1,15037.0,,14111.0,15140.0,17492.0,19079.0,18902.0,,,,,,0.7815,0.6138,,0.5313,,0.8113,0.0974,27700,0.393,0.351,31500,334.876752247489,0.2185867473,,0.2475,Q39624974,Q17203904,Q173,Q29364\\\\\\\\n5,100751,105100,1051,The University of Alabama,Tuscaloosa,AL,www.ua.edu/,financialaid.ua.edu/net-price-calculator/,0,3,4,1,13.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,490.0,610.0,490.0,620.0,480.0,600.0,550.0,555.0,540.0,23.0,31.0,23.0,33.0,22.0,29.0,7.0,8.0,27.0,28.0,26.0,8.0,1202.0,1202.0,0.0,0.0039,0.0,0.0042,0.102,0.0,0.0098,0.0,0.0782,0.1036,0.0,0.0057,0.0692,0.0,0.0115,0.0,0.0,0.0338,0.009000000000000001,0.0,0.0206,0.0,0.0031,0.0,0.0115,0.0,0.036000000000000004,0.0263,0.0109,0.0362,0.0,0.0,0.0,0.0,0.026000000000000002,0.0988,0.2879,0.0118,0.0,31663.0,0.7841,0.1037,0.0437,0.0118,0.0036,0.0009,0.0297,0.0192,0.0033,0.0819,1,21676.0,,18686.0,20013.0,22425.0,23666.0,24578.0,,,,,,0.1938,0.8637,,0.4308,,0.4007,0.081,44500,0.695,0.679,23290,247.596176502985,0.6019442985,,0.6793,Q39625107,Q17204328,Q173,Q79580\\\\\\\\n6,100760,100700,1007,Central Alabama Community College,Alexander City,AL,www.cacc.edu,www.cacc.edu/NetPriceCalculator/14-15/npcalc.html,0,2,2,1,32.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,,,0.0,0.0,0.0,0.0,0.0,0.0,0.0266,0.0082,0.0,0.0,0.1025,0.0,0.0,0.0,0.0,0.2787,0.0,0.0,0.0,0.0,0.0287,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0307,0.3176,0.0,0.0,0.1209,0.0861,0.0,0.0,1492.0,0.6877,0.2802,0.0127,0.002,0.004,0.0007,0.0067,0.002,0.004,0.3733,1,9128.0,,8882.0,8647.0,11681.0,11947.0,13868.0,,,,,,0.5109,,0.5666,,0.4554,0.3234,0.263,27700,0.466,0.395,9500,100.994576074639,0.2510056315,0.2136,,Q39625150,Q17203916,Q173,Q79663\\\\\\\\n7,100812,100800,1008,Athens State University,Athens,AL,www.athens.edu,https://24.athens.edu/apex/prod8/f?p=174:1:3941357449598491,0,3,3,1,31.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,,,0.0,0.0,0.0,0.0,0.0,0.0,0.0462,0.0,0.2192,0.0,0.0,0.0,0.0,0.0,0.0346,0.0538,0.0,0.0231,0.0205,0.0,0.0154,0.0154,0.0038,0.0,0.0026,0.0,0.0308,0.0282,0.0,0.0218,0.0,0.0,0.0,0.0,0.0256,0.0064,0.4449,0.0077,0.0,2888.0,0.7784,0.125,0.0215,0.0076,0.0142,0.001,0.0187,0.001,0.0325,0.5817,1,,,,,,,,,,,,,0.4219,,,,,0.6455,0.6774,38700,0.653,0.612,18000,191.358144141422,0.5038167939,,,Q39625389,Q17203920,Q173,Q203263\\\\\\\\n8,100830,831000,8310,Auburn University at Montgomery,Montgomery,AL,www.aum.edu,www.aum.edu/current-students/financial-information/net-price-calculator,0,3,4,1,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,435.0,495.0,445.0,495.0,,,465.0,470.0,,19.0,24.0,19.0,24.0,17.0,22.0,,,22.0,22.0,20.0,,1009.0,1009.0,0.0,0.02,0.0,0.0,0.0601,0.0,0.0,0.0,0.0584,0.0,0.0,0.0033,0.0,0.0,0.0067,0.0117,0.0,0.0534,0.0083,0.0,0.0,0.0501,0.0,0.0,0.015,0.0,0.0668,0.0351,0.0,0.0401,0.0,0.0,0.0,0.0,0.0267,0.2621,0.2705,0.0117,0.0,4171.0,0.5126,0.3627,0.0141,0.0247,0.006,0.001,0.0319,0.0412,0.0058,0.2592,1,15053.0,,13480.0,14114.0,16829.0,17950.0,17022.0,,,,,,0.4405,0.6566,,0.4766,,0.5565,0.2257,33300,0.616,0.546,23363,248.372240087558,0.4418886199,,0.2207,Q39625474,Q17613566,Q173,Q29364\\\\\\\\n9,100858,100900,1009,Auburn University,Auburn,AL,www.auburn.edu,https://www.auburn.edu/admissions/netpricecalc/freshman.html,0,3,4,1,13.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,530.0,620.0,530.0,640.0,520.0,620.0,575.0,585.0,570.0,24.0,30.0,25.0,32.0,23.0,28.0,7.0,8.0,27.0,29.0,26.0,8.0,1217.0,1217.0,0.0437,0.0133,0.0226,0.0,0.0575,0.0,0.0079,0.0,0.0941,0.1873,0.0,0.0097,0.0337,0.0,0.0088,0.0,0.0,0.0724,0.0097,0.0,0.0267,0.0,0.0014,0.0,0.0093,0.0,0.033,0.0,0.0179,0.0312,0.0,0.0,0.0,0.0,0.0326,0.0667,0.2113,0.009000000000000001,0.0,22095.0,0.8285,0.0673,0.0335,0.0252,0.0052,0.0003,0.0128,0.0214,0.0059,0.0831,1,21984.0,,15591.0,19655.0,23286.0,24591.0,25402.0,,,,,,0.1532,0.9043,,0.7229,,0.32799999999999996,0.0427,48800,0.741,0.726,21500,228.566672168921,0.7239612977,,0.74,Q39625609,Q17203926,Q173,Q225519\\\\\\\\n10,100937,101200,1012,Birmingham Southern College,Birmingham,AL,www.bsc.edu/,www.bsc.edu/fp/np-calculator.cfm,0,3,3,2,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,71.0,500.0,610.0,490.0,570.0,,,555.0,530.0,,23.0,28.0,22.0,29.0,22.0,26.0,,,26.0,26.0,24.0,,1150.0,1150.0,0.0,0.023,0.0,0.0077,0.0268,0.0,0.0,0.0,0.046,0.0077,0.0,0.0077,0.0,0.0,0.023,0.0,0.0,0.1379,0.0498,0.0,0.0383,0.0,0.0307,0.0,0.0575,0.0,0.0728,0.0,0.0,0.0881,0.0,0.0,0.0,0.0,0.1034,0.0,0.2261,0.0536,0.0,1289.0,0.7921,0.1171,0.0217,0.0489,0.006999999999999999,0.0,0.0109,0.0,0.0023,0.0054,1,,23227.0,,,,,,20815.0,19582.0,23126.0,24161.0,25729.0,0.1888,0.8386,,,,0.4729,0.0141,46700,0.637,0.618,26045,276.88460356463,0.7559912854,,0.6439,,Q17203945,Q173,Q79867\\\\\\\\n\\\\\\\", \\\\\\\"local_storage\\\\\\\": \\\\\\\"/data00/dsbox/datamart/memcache_storage/datasets_cache/794d5f7dcddae86817a10e16ee1aecfa.h5\\\\\\\"}\\\"}, \\\"title\\\": {\\\"xml:lang\\\": \\\"en\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"most recent cohorts scorecard elements csv\\\"}, \\\"keywords\\\": {\\\"datatype\\\": \\\"http://www.w3.org/2001/XMLSchema#string\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"unitid opeid opeid6 instnm city stabbr insturl npcurl hcm2 preddeg highdeg control locale hbcu pbi annhi tribal aanapii hsi nanti menonly womenonly relaffil satvr25 satvr75 satmt25 satmt75 satwr25 satwr75 satvrmid satmtmid satwrmid actcm25 actcm75 acten25 acten75 actmt25 actmt75 actwr25 actwr75 actcmmid actenmid actmtmid actwrmid sat avg sat avg all pcip01 pcip03 pcip04 pcip05 pcip09 pcip10 pcip11 pcip12 pcip13 pcip14 pcip15 pcip16 pcip19 pcip22 pcip23 pcip24 pcip25 pcip26 pcip27 pcip29 pcip30 pcip31 pcip38 pcip39 pcip40 pcip41 pcip42 pcip43 pcip44 pcip45 pcip46 pcip47 pcip48 pcip49 pcip50 pcip51 pcip52 pcip54 distanceonly ugds ugds white ugds black ugds hisp ugds asian ugds aian ugds nhpi ugds 2mor ugds nra ugds unkn pptug ef curroper npt4 pub npt4 priv npt41 pub npt42 pub npt43 pub npt44 pub npt45 pub npt41 priv npt42 priv npt43 priv npt44 priv npt45 priv pctpell ret ft4 pooled supp ret ftl4 pooled supp ret pt4 pooled supp ret ptl4 pooled supp pctfloan ug25abv md earn wne p10 gt 25k p6 gt 28k p6 grad debt mdn supp grad debt mdn10yr supp rpy 3yr rt supp c150 l4 pooled supp c150 4 pooled supp unitid wikidata opeid6 wikidata stabbr wikidata city wikidata\\\"}, \\\"datasetLabel\\\": {\\\"xml:lang\\\": \\\"en\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"D4cb70062-77ed-4097-a486-0b43ffe81463\\\"}, \\\"variableName\\\": {\\\"datatype\\\": \\\"http://www.w3.org/2001/XMLSchema#string\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"INSTNM\\\"}, \\\"score\\\": {\\\"datatype\\\": \\\"http://www.w3.org/2001/XMLSchema#double\\\", \\\"type\\\": \\\"literal\\\", \\\"value\\\": \\\"0.9050571841886983\\\"}}, \\\"query_json\\\": {\\\"keywords\\\": [\\\"INSTNM\\\"], \\\"variables\\\": {\\\"INSTNM\\\": \\\"1 10 2 2017 4 410 5 6 a abbey abdill abilene abington abraham academic academie academy accelerated accountancy acupuncture ad adams adelphi adirondack administration adolphus adrian adult advanced advancement advancing adventist adventista advertising aeronautical aeronautics afb age agnes agricultural agriculture aguadilla ai aiken ailano aims air akron alabama alameda alamogordo alamos alaska albany albert albertus albion albizu albright albuquerque alcorn alderson alexandria alfred alice all allan allegany allegheny allen allentown alliance alliant allied alma aloysius alpena altamonte alternative altierus alto altoona alvernia alverno alvin amarillo ambler amboy ambrose america american ameritech ames amherst amityville amridge ana anaheim anchorage ancilla and anderson andrew andrews angeles angelo animal ann anna anne anoka anschutz anselm antelope anthony antillas antioch antonelli antonio anza apex apollo appalachian appleton applied appling aquinas arapahoe arbor arcadia architectural architecture area arecibo argosy aria arizona arkansas arlington armstrong arrow art arte arthur artistic arts arundel asa asbury ash asher asheville ashford ashland ashtabula asm asnuntuck assemblies assist assistants associated assumption at ata atep athens atlanta atlantic auburn audio augsburg augusta augustana augustine aultman aurora austin auto automeca automotive avalon avance ave aveda averett avery aviation avila avondale avtec award ayers azusa b babson bacone bainbridge baja baker bakersfield baldwin ball baltimore banca bancroft bangor baptist barbara barber barbering barclay bard barnard barnes barranquitas barre barrett barry bartending barton baruch basin bastyr batavia bates batesville baton bay bayamon baylor baymeadows bayshore beach beacom beal beau beaufort beaumont beauty beaver beaverton becker beckfield beckley beebe behrend bel bela belanger belhaven bellarmine belle belleville bellevue bellin bellingham bellus belmont beloit beltsville bemidji bend bene benedict benedictine benjamin bennet bennett bennington bentley benton berea bergen berk berkeley berklee berks berkshire bernard bernardino berry beth bethany bethel bethlehem bethune bible biblical big billings biloxi binghamton biola biomedical birmingham bismarck bissonnet bj black blackburn blackhawk blackwood blades blaine blairsville blake bland blauvelt blessing bleu blinn bloomfield bloomington bloomsburg blue bluefield bluegrass bluff bluffton bob boces bodywork bohemia boise bold bon bonaventure bordentown boricua borough bossier boston bothell bottineau boulder bowdoin bowie bowling boylston bradford bradley brainerd bramson branch brandeis brandman brandon brandywine branford braunfels brazosport brecksville brenau brentwood brescia brevard brewton briar briarcliffe brick brickell bridgeport bridgevalley bridgewater brigham brighton brightwood bristol britain brite brittany broadcasting broaddus broadview brockport brockton broken bronx brook brookdale brookfield brookhaven brookline brooklyn broomall broome brothers broward brown brownsburg brownson brownsville brunswick bryan bryant bryn buckeye buckingham bucknell bucks buena buffalo buncombe bunker burlington burnie burnsville burrell business butler butte by c cabarrus cabrillo cabrini caguas cairn cajon calc caldwell calhoun california calumet calvary calvin cambridge camden camelot cameo cameron camp campbell campbellsville campus campuses canada cancer canisius cannella canton canyon canyons cape capella capelli capital capitol capri capstone carbon carbondale cardinal care career careers carey caribbean caribe caribou carl carleton carlos carlow carlsbad carmel carnegie carolina carolinas carrington carroll carrollton carson carsten carteret carthage cary casa casal cascades cascadia case casper castle castleton catawba catherine catholic cattaraugus cazenovia cbd cci cda cecil cedar cedarville cem centenary centennial center centers centerville centra central centre centro centura century cerritos ces cet chabot chadron chamberlain chambersburg chamblee chaminade champaign champlain chandler channel chapel chapman charles charleston charlie charlotte charter charzanne chatfield chatham chattahoochee chattanooga cheeks chemeketa chemung chenoweth cherry chesapeake chester chesterfield chestnut cheyenne cheyney chicago chico chicopee chillicothe chipola chippewa chiropractic choffin chowan christi christian christopher chula cincinnati circus citadel citi cities citizens citrus city clackamas claflin clair claire clara claremont clarendon clarion clark clarke clarks clarksburg clarkson clarksville clary clatsop clayton clear clearfield clearwater cleary clemson clermont cleveland cliff clifton clinic clinical clinton closed cloud clover clovis coachella coalinga coast coastal coastline coba cobb cobleskill coburn cochise cochran coconino cod coe coffeyville cogliano cogswell coker colby coleman colgate colleen college collegeamerica colleges collegiate collin collins colorado colton columbia columbiana columbus comercio commerce commercial commonwealth community company compass computer concept concord concorde concordia conemaugh connecticut connors conservatory consolidated continental continuing converse conway cookeville cookman cooper copiah coppin corazon corban cordon cordova cornell cornerstone corning cornish corona corporation corpus cortiva cortland cosemtology cosmetology costa cosumnes cottey county court covenant covina covington cowley cox coyne cozmo craft crave craven creative creighton crescent crest crestwood criminal crookston cross crosse crossroads crouse crowder crowley crown cruces cruz ct cuesta cuisine culinaire culinary culture culver cumberland cumberlands cuny curry cutler cuyahoga cuyamaca cypress d dabney dade daemen dakota daley dallas dalton dam dame danbury danville darby darlington dartmouth darton davenport david davidson davis davison dawn dawson daymar dayton daytona dba dc dci de deaf dean dearborn decatur defiance degree del delaware delgado delhi delmarva delta denison denmark dental denton denver depaul depauw des desales desert design designers designory detroit devry dewey di diaz dickinson diego diesel digipen digital dillard dimondale directions district divers diversified dividend divinity division dixie dixon dlp doane dodge doheny dominguez dominican dominion don dona donnelly doral dordt dorothy dorsey dothan douglas douglasville dover dow downstate downtown drafting drake dramatic draper drew drexel drive driving drury du dubois dubuque duke dulles duluth dunwoody dupage duquesne durham dutchess dyersburg dynamics e eagan eagle earlham east eastctc eastern eastfield eastlake eastland eastwick eau eberly eckerd ecole ecotech ecpi edgecombe edgewood edic edinboro edison edmonds edp edu education educational educators edward edwardsville ego ehove el elaine elegance elevate elgin elite elizabeth elizabethtown elkhart elkins elley ellsworth elmhurst elmira elmo elms elon embry emerson emmanuel emmaus emory empire employment emporia endicott enfield engineering england enterprise entertainment environmental equestrian erie erlanger erskine esani escanaba escuela essential essex este estelle estes esthetics esthiology estrella eti eugene eunice euphoria eureka european evangel evansville everest everett everglades evergreen evers eves ex excellence excelsior expertise exposito expression expressions extended extension exton f fair fairbanks fairfax fairfield fairleigh fairmont fairview faith fajardo fall falls family fargo faribault farmingdale farmington farms fashion fashions faulkner fayette fayetteville fe fear feather federal federico felician fenton ferris ferrum fiance fidm film findlay fine finger finlandia firelands first fisher fisk fitchburg five flagler flagstaff flathead fletcher flint florence florham florida focus foley folsom fond fontana fontbonne foothill for ford fordham forest forestry forge forks forrest forsyth fort fortis forty fountainhead fox framingham francis franciscan francisco frank franklin frederick fredericksburg fredonia fredric freed fremont fresno friends friendswood front frontier frostburg ft fuld full fullerton fulton funeral furman g gabriel gainesville galen gallaudet gallipolis gallup galveston gannon garden gardena gardens gardner garrett gastonia gate gateway gavilan geauga gene general genesee geneseo genesis geneva george georgetown georgia georgian gerbers german germanna gettysburg gilbert gill girardeau glasgow glen glendale glenville global gloucester god goddard gods gogebic golden goldey goldfarb golf golfers gonzaga good goodwin gordon goshen goucher governors govoha grabber grace graceland graduate grady graham grambling grand grande granger granite grants granville grays grayson great greater greece green greene greenfield greensboro greensburg greenspoint greenville gregory grenada greystone grinnell grossmont groton grove guam guayama guilford gulf gulfport gupton gustavus guy gwinnett gwynedd hackensack hagerstown hair hairdressing hairmasters hairstyling halifax hall hallmark hamilton hamline hammond hampden hampshire hampton hamrick hanceville hancock hands hanford hannah hannibal hanover haras harbor harcum hardeman hardin harding harford harlingen harold harper harris harrisburg harrison harrold harry hartford hartwick harvard harvey hastings hato hattiesburg hauppauge haute haven haverford hawaii hawk hawkeye hays haywood hazard hazleton hds headlines healing health healthcare heart heartland heath heathrow heavilin heidelberg heights helena helene hempstead henager henderson hendrix hennepin henrico henrietta henry heritage herkimer hernando herzing hesperia hesston hialeah hibbing hickey high highland highlands highline hilbert hill hilliard hills hillsboro hillsborough hillyard hilo hinds hiram hiwassee hobart hobby hocking hodges hofstra holistic holland hollins hollywood holmes holy holyoke home homestead hondo hondros honolulu hood hooksett hope hopkins hopkinsville horry hospital hospitality hostos hotelera houghton houma housatonic house houston howard hudson huertas hulman humacao humboldt humphreys hunter huntersville huntingdon huntington huntsville huron hurst hurstborne hussian hutchinson hypnosis ibmc ida idaho ideal illinois images imagine immaculata imperial in inc incarnate independence indian indiana indianapolis indianhead industrial industrialization ingram inland innovation innsbrook institute institutes instituto instruction integral intellitec inter interactive intercoast intercontinental interior intermediate international inver iona iowa ipswich irene irvine irving iselin island islands isle israel italy itasca itawamba ithaca iti ivc ivy j jacinto jackson jacksonville jacobs jamaica james jameson jamestown jarvis jasper jay jean jefferson jeffersonville jenks jenny jersey jessup jesuit jewell jewish jfk jna joaquin john johns johnson johnston johnstown joint jolie joliet jones joplin jordan jose josef joseph josephs joya juan juana juarez judson juilliard junction juniata junior justice kalamazoo kankakee kansas kapiolani kaplan kaskaskia katy kauai kaye kd kean kearney keene keiser kellogg kelly kendall kenilworth kennebec kennedy kenner kennesaw kenneth kenosha kensington kent kentucky kentwood kenyon kettering keuka keys keystone kilgore killeen king kingsborough kingsville kirkwood kirtland kishwaukee kittanning klamath knox knoxville kokomo kootenai kutztown kuyper l la labette laboratory laboure lac lackawanna laconia lady lafayette lagrange laguardia laguna lake lakeland lakes lakeshore lakeview lamar lamoni lamson lancaster land lander landmark landover lane laney langhorne langston lansdale lansing laramie laredo las lasell lassen lauderdale lauderhill laurel laurus law lawrence lawrenceville lawton layton lds le lea leandro learning lebanon lee lees leeward legal lehigh lehman lemoore lenape lenoir leo leon lesley lester letourneau levittown lewis lewiston lewisville lexington liberal liberty liceo licking life lilburn lim lima limestone lincoln linda lindenwood lindsey line linfield linn linthicum linwood lipscomb lithonia little littleton liu liverpool living livingstone livonia llc lloyd location locations lock logan loma lombard london lone long longview longwood lorain loraines loras lord los loudoun louis louisburg louisiana louisville lourdes lowcountry lowell lower loyola lpn lubbock lucie ludlow lufkin luis luke lukes luna luther lutheran luzerne lycoming lyle lynchburg lyndhurst lyndon lynn lynwood lyon lytles m macalester maccormac machias machines machinists mackie macmurray macomb macon madeline madison madisonville madonna magnolia magnus maharishi mahwah main maine mainland maintenance maitland make makeup malcolm malden malone management manassas manatee manati manchester mandl manhattan manhattanville mankato manoa manor mansfield mar maranatha marcos margaret margate maria marian marietta marin marine marion marist maritime marlboro marquette mars marsh marshall marshalltown marti martin martinsburg marty mary marygrove maryland marylhurst marymount maryville marywood mason massachusetts massage massasoit master mateo mattydale maui maumee mawr mayaguez mayfield mayo maysville mayville mcallen mcallister mccann mcconnell mcdaniel mchenry mckendree mckenna mclean mclennan mcminnville mcmurry mcnally mcneese mcpherson mcphs mcrae md mdt meadows mech mechanical mechanics med medaille medanos medford medgar media mediatech medical medicine medrash melbourne mellon melrose memorial memphis mendocino mendota menlo mennonite mentor merced mercer merchandising merchant mercy mercyhurst meredith meridian merrell merrillville merrimack merritt mesa mesabi mesquite messiah metairie methodist metro metropark metropolitan metropolitana mexico mgh miami miat michael michaels michigan mid midamerica middle middlebury middlesex middletown midland midlands midlothian midstate midway midwest midwestern mifflin milan mildred miles military miller millersville millikin mills millsaps milwaukee mineral mines mining minneapolis minnesota minot miracosta miramar misericordia mission missionaries mississippi missoula missouri mitchell mitchells mj moberly mobile model modern modesto mohave mohawk moines mokena moler moline molloy monde monica monmouth monroe monroeville mont montana montcalm montclair monterey montevallo montgomery monticello montreat montrose montserrat moore moorestown moorhead moorpark moraine moravian more morehead morehouse morgan morgantown morningside morrilton morris morrison morrisville morrow morton mortuary moscow motivation motlow motorcycle motoring mott motte mount mountain mountains mountlake mountwest moyne mr mssu mt mti mudd muhlenberg mullins multnomah muncie murfreesboro murphy murray muscle museum music musical muskegon muskingum mycomputercareer myers myomassology myotherapy myrtle n nailcare nails name names nampa namur napa naropa nascar nash nashua nashville nassau natchez national nationwide natural naugatuck navarro nazarene nazareth ne nebraska nelly nelson neosho network networks neumann neumont nevada new newark newberry newbury newington newman newport news newschool newton nhti niagara nicholls nichols nicolet niles noblesville noc non norbert norcross norfolk norman normandale north northampton northcentral northcoast northeast northeastern northern northland northpoint northridge northwest northwestern northwood norwalk norwich nossi notre nova novi ntma nunez nursing nutley nutter nuys nw ny nyack o oahu oak oakland oaks oakton oakwood oberlin obispo occidental occupational occupations ocean oconee odessa oehrlein of ogden ogeechee ogle oglethorpe ohio okc oklahoma olaf old olive olivet olney olympian olympic omaha omnitech on one oneida oneonta online onondaga ontario opportunities oral orange orangeburg oregon oriental orion orlando orleans orlo ort oshkosh osteopathic oswego otero otis ottawa otterbein ottumwa ouachita ouachitas our overland owen owens owensboro owings oxford oxnard ozark ozarks p pa pace pacific paier paine palace palladium palm palmer palo palomar paltz panhandle panola papageorge paradise paramedical paramount paramus paris parish parishes parisian park parker parkersburg parkland parkside parma paroba parsippany partner pasadena pasco paso pass passaic paterson path paul payne pc pci peace peak pearl peay pedro peekskill peirce pellissippi pembroke peninsula penn pennco pennsylvania pensacola peoria pepperdine perimeter permian perry perth peru peter petersburg pfeiffer phagans pharmacy pharr phenix philadelphia philander philip phillips phoenix photography pickaway pickens piedmont piedras pierce pierpont pikes pikeville pillar pima pine pines pineville pinnacle pioneer piscataway pitt pittsburg pittsburgh pittsfield pitzer pj plainfield plains plano platt platteville plattsburgh plaza pleasant plymouth point polaris politecnica polk polytechnic pomeroy pomona pompano ponce pontiac pontifical poplar port portage porter portfolio portland portsmouth post potomac potsdam pottsville poughkeepsie poway powersport practical prairie pratt praxis premiere presbyterian prescott presentation presque pressley prince princeton prism pro production professional professionals professions program programs providence provo public pueblo puerto puget pulaski purchase purdue puyallup q quad queen queens queensborough quest quincy quinn quinnipiac quinsigamond quinta r radford radiation radio radiologic rainey rainy raleigh ramapo ramsey rancho randall randolph range ranger ranken raphael rapid rapids raritan rasmussen ravenna ravenscroft raymore rea reading recording red redlands redwood redwoods reed reedley refrigeration regent regina region regional regis reinhardt remington rend reno rensselaer renton reporting research reserve restaurant resurrection rexburg rey reynolds rhode rhodes rhyne rice richard richardson richey richfield richland richmond rico riddle rider ridge ridgeland ridgewater ridley rieman ringling rio ripon river riverhead rivers riverside rivertown rivier rizzieri road roane roanoke rob robert roberts robeson roche rochelle rochester rock rockford rockhurst rockland rocks rockville rocky roger rogers rogue rolla rollins roman romeoville roosevelt rosa rosalind rose rosedale roseman rosemont roseville ross roswell rouge rowan roxborough roy royale rudae rudy ruidoso rush russel rust rutgers s saber sacramento sacred saddleback sae sage saginaw sagrado sail saint salado salem salina salinas salisbury salish salkehatchie salle salon salt salter salus salve sam samaritan samford samuel san sand sandburg sandusky sandy sanford santa sarah sarasota sargeant sauk savannah sawyer schaumburg schenectady schilling scholars scholastica school schoolcraft schools schreiner schuyler schuylkill science sciences scioto scott scottsdale scranton scripps se seacoast searcy seattle sebastian secours seguin seminary seminole seneca sentara sequoias service services seton setters sewanee seward sewing shadyside shamokin shasta shaw shawnee shear sheen sheffield shelby shelton shenandoah shenango shepherd sheridan shippensburg shoals shore shoreline shorter shreveport shuler siena sierra signature silicon siloam silver simmons simon simpson sinai sinclair sioux siskiyous site six skagit skidmore skilled skills skin skyline skysong slidell slippery smith smiths smyrna snead snow snyder soka solano soledad solutions soma somerset somersworth somerville sonoma sound south southcentral southeast southeastern southern southfield southgate southington southmost southside southtowns southwest southwestern spa spalding sparks spartan spartanburg specs speedway spelman spencerian spokane spring springboro springfield springs sprunt st stafford stage stamford stanbridge stanford stanislaus staples star stark state staten states station stautzenberger steilacoom stephen stephens sterling stetson steuben steubenville steven stevens stevenson stewart stillman stockton stone stonehill stony stout stowe strand stratford stratton strayer street stritch strongsville stroudsburg studies studio stylemaster styling success sues suffolk sul sullivan sum summer summerlin summit sumner sumter sunbury sunset sunstate suny superior support susquehanna sussex sw swarthmore swedish sweet swlr sydney sylvania syracuse system t tabor tacoma takoma talladega tallahassee tampa taos tarleton tarrant taunton taylor tdds tech technical technicians technological technology tecnologia temecula tempe temple tennessee terra terrace terre terry testing teterboro texarkana texas thaddeus the theological theology therapeutic therapy thiel thomas thornton three thunderbird tidewater tiffin tigard tinley tint tioga titusville tobe toccoa toledo tompkins toms toni tougaloo touro town towns towson tractor trade trades traditional trail trailer training transylvania traverse treasure treasury treme trend trevecca tri triangle tribal tribes tricoci trident trine trinidad trinity triton trocaire troy truck truckee trucking truett truman trumbull tualatin tucson tufts tulane tulsa tunxis turabo turlock turnersville tuscarawas tusculum tuskegee tv twin tyler ucas uei ulster ultimate ultrasound umpqua union unit unitech united unity univeristy universal universidad university up upland upper upstate urbana ursinus ursuline utah utica va valdosta valencia valley valparaiso van vance vancouver vanderbilt vanguard vassar vatterott vaughn veeb vega vegas ventura venus vermilion vermont verne vernon vet vici victor victoria view villa villanova vincennes vincent virgin virginia visalia visible vista visual viterbo vocational vocations vogue volunteer voorhees wabash wachusett wade wadena wagner wake walden waldorf wales walla wallace walnut walsh walters warminster warner warren warrendale wartburg warwick washburn washington washtenaw waterbury waterford waterfront waters watkins watsonville waubonsee waukesha wausau wauwatosa way wayland wayne waynesburg weatherford webb webber weber webster weill welch welder welding wellesley wellness wells wellspring wenatchee wentworth weslaco wesley wesleyan west westboro westbrook westbury westchester westech western westfield westminster westmont westport wharton whatcom wheaton wheeling wheelock whelan white whitestone whitewater whitman whittier whitworth wic wichita widener wilberforce wilbur wiley wilkes willamette william williams willingboro williston willoughby wilmington wilson windsor windward wing wingate winona winston winter winthrop wiregrass wisconsin wise wittenberg woburn wofford wolff woman women wood woodbridge woodbury woodland woods woonsocket wooster wor worcester word workforce world worldwide worth worthington wright wynwood wyoming wyotech wytheville x xavier xenon y yakima yale yavapai yeshiva york young youngstown youville yti yuba zane zanesville zona\\\"}, \\\"keywords_search\\\": [\\\"college\\\", \\\"scorecard\\\", \\\"finance\\\", \\\"debt\\\", \\\"earnings\\\"], \\\"variables_search\\\": {}}, \\\"search_type\\\": \\\"general\\\"}, \\\"dataframe_column_names\\\": {\\\"left_names\\\": [\\\"d3mIndex\\\", \\\"UNITID\\\", \\\"INSTNM\\\", \\\"PCTFLOAN\\\", \\\"CONTROL\\\", \\\"STABBR\\\", \\\"PCIP16\\\", \\\"MD_EARN_WNE_P10\\\", \\\"PPTUG_EF\\\", \\\"UGDS_WHITE\\\", \\\"UGDS_BLACK\\\", \\\"UGDS_HISP\\\", \\\"UGDS_ASIAN\\\", \\\"SATMTMID\\\", \\\"SATVRMID\\\", \\\"SATWRMID\\\", \\\"UGDS\\\", \\\"PREDDEG\\\", \\\"DEBT_EARNINGS_RATIO\\\", \\\"STABBR_wikidata\\\"], \\\"right_names\\\": [\\\"UNITID\\\", \\\"OPEID\\\", \\\"OPEID6\\\", \\\"INSTNM\\\", \\\"CITY\\\", \\\"STABBR\\\", \\\"INSTURL\\\", \\\"NPCURL\\\", \\\"HCM2\\\", \\\"PREDDEG\\\", \\\"HIGHDEG\\\", \\\"CONTROL\\\", \\\"LOCALE\\\", \\\"HBCU\\\", \\\"PBI\\\", \\\"ANNHI\\\", \\\"TRIBAL\\\", \\\"AANAPII\\\", \\\"HSI\\\", \\\"NANTI\\\", \\\"MENONLY\\\", \\\"WOMENONLY\\\", \\\"RELAFFIL\\\", \\\"SATVR25\\\", \\\"SATVR75\\\", \\\"SATMT25\\\", \\\"SATMT75\\\", \\\"SATWR25\\\", \\\"SATWR75\\\", \\\"SATVRMID\\\", \\\"SATMTMID\\\", \\\"SATWRMID\\\", \\\"ACTCM25\\\", \\\"ACTCM75\\\", \\\"ACTEN25\\\", \\\"ACTEN75\\\", \\\"ACTMT25\\\", \\\"ACTMT75\\\", \\\"ACTWR25\\\", \\\"ACTWR75\\\", \\\"ACTCMMID\\\", \\\"ACTENMID\\\", \\\"ACTMTMID\\\", \\\"ACTWRMID\\\", \\\"SAT_AVG\\\", \\\"SAT_AVG_ALL\\\", \\\"PCIP01\\\", \\\"PCIP03\\\", \\\"PCIP04\\\", \\\"PCIP05\\\", \\\"PCIP09\\\", \\\"PCIP10\\\", \\\"PCIP11\\\", \\\"PCIP12\\\", \\\"PCIP13\\\", \\\"PCIP14\\\", \\\"PCIP15\\\", \\\"PCIP16\\\", \\\"PCIP19\\\", \\\"PCIP22\\\", \\\"PCIP23\\\", \\\"PCIP24\\\", \\\"PCIP25\\\", \\\"PCIP26\\\", \\\"PCIP27\\\", \\\"PCIP29\\\", \\\"PCIP30\\\", \\\"PCIP31\\\", \\\"PCIP38\\\", \\\"PCIP39\\\", \\\"PCIP40\\\", \\\"PCIP41\\\", \\\"PCIP42\\\", \\\"PCIP43\\\", \\\"PCIP44\\\", \\\"PCIP45\\\", \\\"PCIP46\\\", \\\"PCIP47\\\", \\\"PCIP48\\\", \\\"PCIP49\\\", \\\"PCIP50\\\", \\\"PCIP51\\\", \\\"PCIP52\\\", \\\"PCIP54\\\", \\\"DISTANCEONLY\\\", \\\"UGDS\\\", \\\"UGDS_WHITE\\\", \\\"UGDS_BLACK\\\", \\\"UGDS_HISP\\\", \\\"UGDS_ASIAN\\\", \\\"UGDS_AIAN\\\", \\\"UGDS_NHPI\\\", \\\"UGDS_2MOR\\\", \\\"UGDS_NRA\\\", \\\"UGDS_UNKN\\\", \\\"PPTUG_EF\\\", \\\"CURROPER\\\", \\\"NPT4_PUB\\\", \\\"NPT4_PRIV\\\", \\\"NPT41_PUB\\\", \\\"NPT42_PUB\\\", \\\"NPT43_PUB\\\", \\\"NPT44_PUB\\\", \\\"NPT45_PUB\\\", \\\"NPT41_PRIV\\\", \\\"NPT42_PRIV\\\", \\\"NPT43_PRIV\\\", \\\"NPT44_PRIV\\\", \\\"NPT45_PRIV\\\", \\\"PCTPELL\\\", \\\"RET_FT4_POOLED_SUPP\\\", \\\"RET_FTL4_POOLED_SUPP\\\", \\\"RET_PT4_POOLED_SUPP\\\", \\\"RET_PTL4_POOLED_SUPP\\\", \\\"PCTFLOAN\\\", \\\"UG25ABV\\\", \\\"MD_EARN_WNE_P10\\\", \\\"GT_25K_P6\\\", \\\"GT_28K_P6\\\", \\\"GRAD_DEBT_MDN_SUPP\\\", \\\"GRAD_DEBT_MDN10YR_SUPP\\\", \\\"RPY_3YR_RT_SUPP\\\", \\\"C150_L4_POOLED_SUPP\\\", \\\"C150_4_POOLED_SUPP\\\", \\\"UNITID_wikidata\\\", \\\"OPEID6_wikidata\\\", \\\"STABBR_wikidata\\\", \\\"CITY_wikidata\\\"]}, \\\"augmentation\\\": {\\\"properties\\\": \\\"join\\\", \\\"left_columns\\\": [[2]], \\\"right_columns\\\": [[3]]}, \\\"datamart_type\\\": \\\"isi\\\"}\", \"metadata\": [{\"metadata\": {\"dimension\": {\"length\": 7175, \"name\": \"rows\", \"semantic_types\": [\"https://metadata.datadrivendiscovery.org/types/TabularRow\"]}, \"schema\": \"https://metadata.datadrivendiscovery.org/schemas/v0/container.json\", \"semantic_types\": [\"https://metadata.datadrivendiscovery.org/types/Table\"], \"structural_type\": \"d3m.container.pandas.DataFrame\"}, \"selector\": []}, {\"metadata\": {\"dimension\": {\"length\": 128, \"name\": \"columns\", \"semantic_types\": [\"https://metadata.datadrivendiscovery.org/types/TabularColumn\"]}}, \"selector\": [\"__ALL_ELEMENTS__\"]}, {\"metadata\": {\"name\": \"UNITID\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 0]}, {\"metadata\": {\"name\": \"OPEID\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 1]}, {\"metadata\": {\"name\": \"OPEID6\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 2]}, {\"metadata\": {\"name\": \"INSTNM\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 3]}, {\"metadata\": {\"name\": \"CITY\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 4]}, {\"metadata\": {\"name\": \"STABBR\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 5]}, {\"metadata\": {\"name\": \"INSTURL\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 6]}, {\"metadata\": {\"name\": \"NPCURL\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 7]}, {\"metadata\": {\"name\": \"HCM2\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 8]}, {\"metadata\": {\"name\": \"PREDDEG\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 9]}, {\"metadata\": {\"name\": \"HIGHDEG\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 10]}, {\"metadata\": {\"name\": \"CONTROL\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 11]}, {\"metadata\": {\"name\": \"LOCALE\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 12]}, {\"metadata\": {\"name\": \"HBCU\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 13]}, {\"metadata\": {\"name\": \"PBI\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 14]}, {\"metadata\": {\"name\": \"ANNHI\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 15]}, {\"metadata\": {\"name\": \"TRIBAL\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 16]}, {\"metadata\": {\"name\": \"AANAPII\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 17]}, {\"metadata\": {\"name\": \"HSI\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 18]}, {\"metadata\": {\"name\": \"NANTI\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 19]}, {\"metadata\": {\"name\": \"MENONLY\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 20]}, {\"metadata\": {\"name\": \"WOMENONLY\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 21]}, {\"metadata\": {\"name\": \"RELAFFIL\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 22]}, {\"metadata\": {\"name\": \"SATVR25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 23]}, {\"metadata\": {\"name\": \"SATVR75\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 24]}, {\"metadata\": {\"name\": \"SATMT25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 25]}, {\"metadata\": {\"name\": \"SATMT75\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 26]}, {\"metadata\": {\"name\": \"SATWR25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 27]}, {\"metadata\": {\"name\": \"SATWR75\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 28]}, {\"metadata\": {\"name\": \"SATVRMID\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 29]}, {\"metadata\": {\"name\": \"SATMTMID\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 30]}, {\"metadata\": {\"name\": \"SATWRMID\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 31]}, {\"metadata\": {\"name\": \"ACTCM25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 32]}, {\"metadata\": {\"name\": \"ACTCM75\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 33]}, {\"metadata\": {\"name\": \"ACTEN25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 34]}, {\"metadata\": {\"name\": \"ACTEN75\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 35]}, {\"metadata\": {\"name\": \"ACTMT25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 36]}, {\"metadata\": {\"name\": \"ACTMT75\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 37]}, {\"metadata\": {\"name\": \"ACTWR25\", \"semantic_types\": [\"https://metadata.datadrivendiscovery.org/types/CategoricalData\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 38]}, {\"metadata\": {\"name\": \"ACTWR75\", \"semantic_types\": [\"https://metadata.datadrivendiscovery.org/types/CategoricalData\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 39]}, {\"metadata\": {\"name\": \"ACTCMMID\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 40]}, {\"metadata\": {\"name\": \"ACTENMID\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 41]}, {\"metadata\": {\"name\": \"ACTMTMID\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 42]}, {\"metadata\": {\"name\": \"ACTWRMID\", \"semantic_types\": [\"https://metadata.datadrivendiscovery.org/types/CategoricalData\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 43]}, {\"metadata\": {\"name\": \"SAT_AVG\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 44]}, {\"metadata\": {\"name\": \"SAT_AVG_ALL\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 45]}, {\"metadata\": {\"name\": \"PCIP01\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 46]}, {\"metadata\": {\"name\": \"PCIP03\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 47]}, {\"metadata\": {\"name\": \"PCIP04\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 48]}, {\"metadata\": {\"name\": \"PCIP05\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 49]}, {\"metadata\": {\"name\": \"PCIP09\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 50]}, {\"metadata\": {\"name\": \"PCIP10\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 51]}, {\"metadata\": {\"name\": \"PCIP11\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 52]}, {\"metadata\": {\"name\": \"PCIP12\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 53]}, {\"metadata\": {\"name\": \"PCIP13\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 54]}, {\"metadata\": {\"name\": \"PCIP14\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 55]}, {\"metadata\": {\"name\": \"PCIP15\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 56]}, {\"metadata\": {\"name\": \"PCIP16\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 57]}, {\"metadata\": {\"name\": \"PCIP19\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 58]}, {\"metadata\": {\"name\": \"PCIP22\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 59]}, {\"metadata\": {\"name\": \"PCIP23\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 60]}, {\"metadata\": {\"name\": \"PCIP24\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 61]}, {\"metadata\": {\"name\": \"PCIP25\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 62]}, {\"metadata\": {\"name\": \"PCIP26\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 63]}, {\"metadata\": {\"name\": \"PCIP27\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 64]}, {\"metadata\": {\"name\": \"PCIP29\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 65]}, {\"metadata\": {\"name\": \"PCIP30\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 66]}, {\"metadata\": {\"name\": \"PCIP31\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 67]}, {\"metadata\": {\"name\": \"PCIP38\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 68]}, {\"metadata\": {\"name\": \"PCIP39\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 69]}, {\"metadata\": {\"name\": \"PCIP40\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 70]}, {\"metadata\": {\"name\": \"PCIP41\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 71]}, {\"metadata\": {\"name\": \"PCIP42\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 72]}, {\"metadata\": {\"name\": \"PCIP43\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 73]}, {\"metadata\": {\"name\": \"PCIP44\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 74]}, {\"metadata\": {\"name\": \"PCIP45\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 75]}, {\"metadata\": {\"name\": \"PCIP46\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 76]}, {\"metadata\": {\"name\": \"PCIP47\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 77]}, {\"metadata\": {\"name\": \"PCIP48\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 78]}, {\"metadata\": {\"name\": \"PCIP49\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 79]}, {\"metadata\": {\"name\": \"PCIP50\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 80]}, {\"metadata\": {\"name\": \"PCIP51\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 81]}, {\"metadata\": {\"name\": \"PCIP52\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 82]}, {\"metadata\": {\"name\": \"PCIP54\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 83]}, {\"metadata\": {\"name\": \"DISTANCEONLY\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 84]}, {\"metadata\": {\"name\": \"UGDS\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 85]}, {\"metadata\": {\"name\": \"UGDS_WHITE\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 86]}, {\"metadata\": {\"name\": \"UGDS_BLACK\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 87]}, {\"metadata\": {\"name\": \"UGDS_HISP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 88]}, {\"metadata\": {\"name\": \"UGDS_ASIAN\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 89]}, {\"metadata\": {\"name\": \"UGDS_AIAN\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 90]}, {\"metadata\": {\"name\": \"UGDS_NHPI\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 91]}, {\"metadata\": {\"name\": \"UGDS_2MOR\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 92]}, {\"metadata\": {\"name\": \"UGDS_NRA\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 93]}, {\"metadata\": {\"name\": \"UGDS_UNKN\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 94]}, {\"metadata\": {\"name\": \"PPTUG_EF\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 95]}, {\"metadata\": {\"name\": \"CURROPER\", \"semantic_types\": [\"http://schema.org/Integer\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 96]}, {\"metadata\": {\"name\": \"NPT4_PUB\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 97]}, {\"metadata\": {\"name\": \"NPT4_PRIV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 98]}, {\"metadata\": {\"name\": \"NPT41_PUB\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 99]}, {\"metadata\": {\"name\": \"NPT42_PUB\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 100]}, {\"metadata\": {\"name\": \"NPT43_PUB\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 101]}, {\"metadata\": {\"name\": \"NPT44_PUB\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 102]}, {\"metadata\": {\"name\": \"NPT45_PUB\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 103]}, {\"metadata\": {\"name\": \"NPT41_PRIV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 104]}, {\"metadata\": {\"name\": \"NPT42_PRIV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 105]}, {\"metadata\": {\"name\": \"NPT43_PRIV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 106]}, {\"metadata\": {\"name\": \"NPT44_PRIV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 107]}, {\"metadata\": {\"name\": \"NPT45_PRIV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 108]}, {\"metadata\": {\"name\": \"PCTPELL\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 109]}, {\"metadata\": {\"name\": \"RET_FT4_POOLED_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 110]}, {\"metadata\": {\"name\": \"RET_FTL4_POOLED_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 111]}, {\"metadata\": {\"name\": \"RET_PT4_POOLED_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 112]}, {\"metadata\": {\"name\": \"RET_PTL4_POOLED_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 113]}, {\"metadata\": {\"name\": \"PCTFLOAN\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 114]}, {\"metadata\": {\"name\": \"UG25ABV\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 115]}, {\"metadata\": {\"name\": \"MD_EARN_WNE_P10\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 116]}, {\"metadata\": {\"name\": \"GT_25K_P6\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 117]}, {\"metadata\": {\"name\": \"GT_28K_P6\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 118]}, {\"metadata\": {\"name\": \"GRAD_DEBT_MDN_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 119]}, {\"metadata\": {\"name\": \"GRAD_DEBT_MDN10YR_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 120]}, {\"metadata\": {\"name\": \"RPY_3YR_RT_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 121]}, {\"metadata\": {\"name\": \"C150_L4_POOLED_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 122]}, {\"metadata\": {\"name\": \"C150_4_POOLED_SUPP\", \"semantic_types\": [\"http://schema.org/Float\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 123]}, {\"metadata\": {\"name\": \"UNITID_wikidata\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 124]}, {\"metadata\": {\"name\": \"OPEID6_wikidata\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 125]}, {\"metadata\": {\"name\": \"STABBR_wikidata\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 126]}, {\"metadata\": {\"name\": \"CITY_wikidata\", \"semantic_types\": [\"http://schema.org/Text\", \"https://metadata.datadrivendiscovery.org/types/Attribute\"], \"structural_type\": \"str\"}, \"selector\": [\"__ALL_ELEMENTS__\", 127]}], \"preview_only\": false, \"sample\": \"UNITID,OPEID,OPEID6,INSTNM,CITY,STABBR,INSTURL,NPCURL,HCM2,PREDDEG,HIGHDEG,CONTROL,LOCALE,HBCU,PBI,ANNHI,TRIBAL,AANAPII,HSI,NANTI,MENONLY,WOMENONLY,RELAFFIL,SATVR25,SATVR75,SATMT25,SATMT75,SATWR25,SATWR75,SATVRMID,SATMTMID,SATWRMID,ACTCM25,ACTCM75,ACTEN25,ACTEN75,ACTMT25,ACTMT75,ACTWR25,ACTWR75,ACTCMMID,ACTENMID,ACTMTMID,ACTWRMID,SAT_AVG,SAT_AVG_ALL,PCIP01,PCIP03,PCIP04,PCIP05,PCIP09,PCIP10,PCIP11,PCIP12,PCIP13,PCIP14,PCIP15,PCIP16,PCIP19,PCIP22,PCIP23,PCIP24,PCIP25,PCIP26,PCIP27,PCIP29,PCIP30,PCIP31,PCIP38,PCIP39,PCIP40,PCIP41,PCIP42,PCIP43,PCIP44,PCIP45,PCIP46,PCIP47,PCIP48,PCIP49,PCIP50,PCIP51,PCIP52,PCIP54,DISTANCEONLY,UGDS,UGDS_WHITE,UGDS_BLACK,UGDS_HISP,UGDS_ASIAN,UGDS_AIAN,UGDS_NHPI,UGDS_2MOR,UGDS_NRA,UGDS_UNKN,PPTUG_EF,CURROPER,NPT4_PUB,NPT4_PRIV,NPT41_PUB,NPT42_PUB,NPT43_PUB,NPT44_PUB,NPT45_PUB,NPT41_PRIV,NPT42_PRIV,NPT43_PRIV,NPT44_PRIV,NPT45_PRIV,PCTPELL,RET_FT4_POOLED_SUPP,RET_FTL4_POOLED_SUPP,RET_PT4_POOLED_SUPP,RET_PTL4_POOLED_SUPP,PCTFLOAN,UG25ABV,MD_EARN_WNE_P10,GT_25K_P6,GT_28K_P6,GRAD_DEBT_MDN_SUPP,GRAD_DEBT_MDN10YR_SUPP,RPY_3YR_RT_SUPP,C150_L4_POOLED_SUPP,C150_4_POOLED_SUPP,UNITID_wikidata,OPEID6_wikidata,STABBR_wikidata,CITY_wikidata\\n100654,100200,1002,Alabama A & M University,Normal,AL,www.aamu.edu/,www2.aamu.edu/scripts/netpricecalc/npcalc.htm,0,3,4,1,12.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,380.0,470.0,370.0,470.0,370.0,457.0,425.0,420.0,414.0,16.0,19.0,14.0,20.0,15.0,18.0,,,18.0,17.0,17.0,,849.0,849.0,0.0448,0.0142,0.0071,0.0,0.0,0.0354,0.0401,0.0,0.1132,0.0896,0.0472,0.0,0.033,0.0,0.0094,0.066,0.0,0.0708,0.0024,0.0,0.0,0.0,0.0,0.0,0.0307,0.0,0.0472,0.0519,0.0377,0.0448,0.0,0.0,0.0,0.0,0.0283,0.0,0.1863,0.0,0.0,4616.0,0.0256,0.9129,0.0076,0.0019,0.0024,0.0017,0.0401,0.0065,0.0013,0.0877,1,15567.0,,15043.0,15491.0,17335.0,19562.0,18865.0,,,,,,0.7039,0.5774,,0.309,,0.7667,0.0859,31000,0.45299999999999996,0.431,32750,348.16551225731,0.2531554273,,0.2913,Q39624632,Q17203888,Q173,Q575407\\n100663,105200,1052,University of Alabama at Birmingham,Birmingham,AL,www.uab.edu,uab.studentaidcalculator.com/survey.aspx,0,3,4,1,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,480.0,640.0,490.0,660.0,,,560.0,575.0,,21.0,28.0,22.0,30.0,19.0,26.0,,,25.0,26.0,23.0,,1125.0,1125.0,0.0,0.0,0.0,0.0005,0.036000000000000004,0.0,0.0131,0.0,0.0748,0.0599,0.0,0.0059,0.0,0.0,0.0158,0.0135,0.0,0.0734,0.009000000000000001,0.0,0.0,0.0,0.005,0.0,0.0212,0.0,0.0766,0.0243,0.0221,0.0365,0.0,0.0,0.0,0.0,0.0392,0.25,0.2072,0.0162,0.0,12047.0,0.5786,0.2626,0.0309,0.0598,0.0028,0.0004,0.0387,0.0179,0.0083,0.2578,1,16475.0,,13849.0,15385.0,18022.0,18705.0,19319.0,,,,,,0.3525,0.8007,,0.5178,,0.5179,0.2363,41200,0.669,0.631,21833,232.106797835537,0.5139631610000001,,0.5384,Q39624677,Q17204336,Q173,Q79867\\n100690,2503400,25034,Amridge University,Montgomery,AL,www.amridgeuniversity.edu,www2.amridgeuniversity.edu:9091/,0,3,4,2,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,74.0,,,,,,,,,,,,,,,,,,,,,,,,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0889,0.0,0.0,0.0889,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4,0.0,0.0,0.0,0.0667,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3556,0.0,1.0,293.0,0.157,0.2355,0.0068,0.0,0.0,0.0034,0.0,0.0,0.5973,0.5392,1,,10155.0,,,,,,10155.0,,,,,0.6971,PrivacySuppressed,,PrivacySuppressed,,0.8436,0.8571,39600,0.6579999999999999,0.542,22890,243.34377329984198,0.23076923079999997,,PrivacySuppressed,Q39624831,Q17337864,Q173,Q29364\\n100706,105500,1055,University of Alabama in Huntsville,Huntsville,AL,www.uah.edu,finaid.uah.edu/,0,3,4,1,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,520.0,660.0,540.0,680.0,,,590.0,610.0,,25.0,31.0,24.0,33.0,23.0,29.0,,,28.0,29.0,26.0,,1257.0,1257.0,0.0,0.0,0.0,0.0,0.0301,0.0,0.0499,0.0,0.0282,0.2702,0.0,0.0151,0.0,0.0,0.0122,0.0,0.0,0.0603,0.0132,0.0,0.0,0.0,0.0113,0.0,0.0226,0.0,0.016,0.0,0.0,0.0188,0.0,0.0,0.0,0.0,0.0264,0.1911,0.225,0.0094,0.0,6346.0,0.7148,0.1131,0.0411,0.0414,0.012,0.0,0.0181,0.0303,0.0292,0.1746,1,19423.0,,15971.0,18016.0,20300.0,21834.0,22059.0,,,,,,0.2949,0.8161,,0.5116,,0.4312,0.2255,46700,0.685,0.649,22647,240.76043835393298,0.5485090297999999,,0.4905,Q39624901,Q17204354,Q173,Q79860\\n100724,100500,1005,Alabama State University,Montgomery,AL,www.alasu.edu,www.alasu.edu/cost-aid/forms/calculator/index.aspx,0,3,4,1,12.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,370.0,460.0,360.0,460.0,,,415.0,410.0,,15.0,19.0,14.0,19.0,15.0,17.0,,,17.0,17.0,16.0,,825.0,825.0,0.0,0.0,0.0,0.0,0.1023,0.0,0.0503,0.0,0.1364,0.0,0.0,0.0,0.0,0.0,0.0114,0.0,0.0,0.0779,0.0146,0.0,0.0,0.0211,0.0,0.0,0.0244,0.0,0.0503,0.1412,0.0633,0.013,0.0,0.0,0.0,0.0,0.0487,0.1429,0.0974,0.0049,0.0,4704.0,0.0138,0.9337,0.0111,0.0028,0.0013,0.0004,0.0111,0.0159,0.01,0.0727,1,15037.0,,14111.0,15140.0,17492.0,19079.0,18902.0,,,,,,0.7815,0.6138,,0.5313,,0.8113,0.0974,27700,0.39299999999999996,0.35100000000000003,31500,334.87675224748904,0.2185867473,,0.2475,Q39624974,Q17203904,Q173,Q29364\\n100751,105100,1051,The University of Alabama,Tuscaloosa,AL,www.ua.edu/,financialaid.ua.edu/net-price-calculator/,0,3,4,1,13.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,490.0,610.0,490.0,620.0,480.0,600.0,550.0,555.0,540.0,23.0,31.0,23.0,33.0,22.0,29.0,7.0,8.0,27.0,28.0,26.0,8.0,1202.0,1202.0,0.0,0.0039,0.0,0.0042,0.102,0.0,0.0098,0.0,0.0782,0.1036,0.0,0.0057,0.0692,0.0,0.0115,0.0,0.0,0.0338,0.009000000000000001,0.0,0.0206,0.0,0.0031,0.0,0.0115,0.0,0.036000000000000004,0.0263,0.0109,0.0362,0.0,0.0,0.0,0.0,0.026,0.0988,0.2879,0.0118,0.0,31663.0,0.7841,0.1037,0.0437,0.0118,0.0036,0.0009,0.0297,0.0192,0.0033,0.0819,1,21676.0,,18686.0,20013.0,22425.0,23666.0,24578.0,,,,,,0.1938,0.8637,,0.4308,,0.4007,0.081,44500,0.695,0.679,23290,247.596176502985,0.6019442985,,0.6793,Q39625107,Q17204328,Q173,Q79580\\n100760,100700,1007,Central Alabama Community College,Alexander City,AL,www.cacc.edu,www.cacc.edu/NetPriceCalculator/14-15/npcalc.html,0,2,2,1,32.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,,,0.0,0.0,0.0,0.0,0.0,0.0,0.0266,0.0082,0.0,0.0,0.1025,0.0,0.0,0.0,0.0,0.2787,0.0,0.0,0.0,0.0,0.0287,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0307,0.3176,0.0,0.0,0.1209,0.0861,0.0,0.0,1492.0,0.6877,0.2802,0.0127,0.002,0.004,0.0007,0.0067,0.002,0.004,0.3733,1,9128.0,,8882.0,8647.0,11681.0,11947.0,13868.0,,,,,,0.5109,,0.5666,,0.4554,0.3234,0.263,27700,0.466,0.395,9500,100.994576074639,0.2510056315,0.2136,,Q39625150,Q17203916,Q173,Q79663\\n100812,100800,1008,Athens State University,Athens,AL,www.athens.edu,https://24.athens.edu/apex/prod8/f?p=174:1:3941357449598491,0,3,3,1,31.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,,,0.0,0.0,0.0,0.0,0.0,0.0,0.0462,0.0,0.2192,0.0,0.0,0.0,0.0,0.0,0.0346,0.0538,0.0,0.0231,0.0205,0.0,0.0154,0.0154,0.0038,0.0,0.0026,0.0,0.0308,0.0282,0.0,0.0218,0.0,0.0,0.0,0.0,0.0256,0.0064,0.4449,0.0077,0.0,2888.0,0.7784,0.125,0.0215,0.0076,0.0142,0.001,0.0187,0.001,0.0325,0.5817,1,,,,,,,,,,,,,0.4219,,,,,0.6455,0.6774,38700,0.653,0.612,18000,191.358144141422,0.5038167939,,,Q39625389,Q17203920,Q173,Q203263\\n100830,831000,8310,Auburn University at Montgomery,Montgomery,AL,www.aum.edu,www.aum.edu/current-students/financial-information/net-price-calculator,0,3,4,1,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,435.0,495.0,445.0,495.0,,,465.0,470.0,,19.0,24.0,19.0,24.0,17.0,22.0,,,22.0,22.0,20.0,,1009.0,1009.0,0.0,0.02,0.0,0.0,0.0601,0.0,0.0,0.0,0.0584,0.0,0.0,0.0033,0.0,0.0,0.0067,0.0117,0.0,0.0534,0.0083,0.0,0.0,0.0501,0.0,0.0,0.015,0.0,0.0668,0.0351,0.0,0.0401,0.0,0.0,0.0,0.0,0.0267,0.2621,0.2705,0.0117,0.0,4171.0,0.5126,0.3627,0.0141,0.0247,0.006,0.001,0.0319,0.0412,0.0058,0.2592,1,15053.0,,13480.0,14114.0,16829.0,17950.0,17022.0,,,,,,0.4405,0.6566,,0.4766,,0.5565,0.2257,33300,0.616,0.546,23363,248.37224008755797,0.4418886199,,0.2207,Q39625474,Q17613566,Q173,Q29364\\n100858,100900,1009,Auburn University,Auburn,AL,www.auburn.edu,https://www.auburn.edu/admissions/netpricecalc/freshman.html,0,3,4,1,13.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,,530.0,620.0,530.0,640.0,520.0,620.0,575.0,585.0,570.0,24.0,30.0,25.0,32.0,23.0,28.0,7.0,8.0,27.0,29.0,26.0,8.0,1217.0,1217.0,0.0437,0.0133,0.0226,0.0,0.0575,0.0,0.0079,0.0,0.0941,0.1873,0.0,0.0097,0.0337,0.0,0.0088,0.0,0.0,0.0724,0.0097,0.0,0.0267,0.0,0.0014,0.0,0.0093,0.0,0.033,0.0,0.0179,0.0312,0.0,0.0,0.0,0.0,0.0326,0.0667,0.2113,0.009000000000000001,0.0,22095.0,0.8285,0.0673,0.0335,0.0252,0.0052,0.0003,0.0128,0.0214,0.0059,0.0831,1,21984.0,,15591.0,19655.0,23286.0,24591.0,25402.0,,,,,,0.1532,0.9043,,0.7229,,0.328,0.0427,48800,0.741,0.726,21500,228.566672168921,0.7239612977,,0.74,Q39625609,Q17203926,Q173,Q225519\\n100937,101200,1012,Birmingham Southern College,Birmingham,AL,www.bsc.edu/,www.bsc.edu/fp/np-calculator.cfm,0,3,3,2,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,71.0,500.0,610.0,490.0,570.0,,,555.0,530.0,,23.0,28.0,22.0,29.0,22.0,26.0,,,26.0,26.0,24.0,,1150.0,1150.0,0.0,0.023,0.0,0.0077,0.0268,0.0,0.0,0.0,0.046,0.0077,0.0,0.0077,0.0,0.0,0.023,0.0,0.0,0.1379,0.0498,0.0,0.0383,0.0,0.0307,0.0,0.0575,0.0,0.0728,0.0,0.0,0.0881,0.0,0.0,0.0,0.0,0.1034,0.0,0.2261,0.0536,0.0,1289.0,0.7921,0.1171,0.0217,0.0489,0.006999999999999999,0.0,0.0109,0.0,0.0023,0.0054,1,,23227.0,,,,,,20815.0,19582.0,23126.0,24161.0,25729.0,0.1888,0.8386,,,,0.4729,0.0141,46700,0.637,0.618,26045,276.88460356462997,0.7559912854,,0.6439,,Q17203945,Q173,Q79867\\n\", \"score\": 0.9050571841886983, \"summary\": {\"Columns\": [\"[0] UNITID\", \"[1] OPEID\", \"[2] OPEID6\", \"[3] INSTNM\", \"[4] CITY\", \"[5] STABBR\", \"[6] INSTURL\", \"[7] NPCURL\", \"[8] HCM2\", \"[9] PREDDEG\", \"[10] HIGHDEG\", \"[11] CONTROL\", \"[12] LOCALE\", \"[13] HBCU\", \"[14] PBI\", \"[15] ANNHI\", \"[16] TRIBAL\", \"[17] AANAPII\", \"[18] HSI\", \"[19] NANTI\", \"[20] MENONLY\", \"[21] WOMENONLY\", \"[22] RELAFFIL\", \"[23] SATVR25\", \"[24] SATVR75\", \"[25] SATMT25\", \"[26] SATMT75\", \"[27] SATWR25\", \"[28] SATWR75\", \"[29] SATVRMID\", \"[30] SATMTMID\", \"[31] SATWRMID\", \"[32] ACTCM25\", \"[33] ACTCM75\", \"[34] ACTEN25\", \"[35] ACTEN75\", \"[36] ACTMT25\", \"[37] ACTMT75\", \"[38] ACTWR25\", \"[39] ACTWR75\", \"[40] ACTCMMID\", \"[41] ACTENMID\", \"[42] ACTMTMID\", \"[43] ACTWRMID\", \"[44] SAT_AVG\", \"[45] SAT_AVG_ALL\", \"[46] PCIP01\", \"[47] PCIP03\", \"[48] PCIP04\", \"[49] PCIP05\", \"[50] PCIP09\", \"[51] PCIP10\", \"[52] PCIP11\", \"[53] PCIP12\", \"[54] PCIP13\", \"[55] PCIP14\", \"[56] PCIP15\", \"[57] PCIP16\", \"[58] PCIP19\", \"[59] PCIP22\", \"[60] PCIP23\", \"[61] PCIP24\", \"[62] PCIP25\", \"[63] PCIP26\", \"[64] PCIP27\", \"[65] PCIP29\", \"[66] PCIP30\", \"[67] PCIP31\", \"[68] PCIP38\", \"[69] PCIP39\", \"[70] PCIP40\", \"[71] PCIP41\", \"[72] PCIP42\", \"[73] PCIP43\", \"[74] PCIP44\", \"[75] PCIP45\", \"[76] PCIP46\", \"[77] PCIP47\", \"[78] PCIP48\", \"[79] PCIP49\", \"[80] PCIP50\", \"[81] PCIP51\", \"[82] PCIP52\", \"[83] PCIP54\", \"[84] DISTANCEONLY\", \"[85] UGDS\", \"[86] UGDS_WHITE\", \"[87] UGDS_BLACK\", \"[88] UGDS_HISP\", \"[89] UGDS_ASIAN\", \"[90] UGDS_AIAN\", \"[91] UGDS_NHPI\", \"[92] UGDS_2MOR\", \"[93] UGDS_NRA\", \"[94] UGDS_UNKN\", \"[95] PPTUG_EF\", \"[96] CURROPER\", \"[97] NPT4_PUB\", \"[98] NPT4_PRIV\", \"[99] NPT41_PUB\", \"[100] NPT42_PUB\", \"[101] NPT43_PUB\", \"[102] NPT44_PUB\", \"[103] NPT45_PUB\", \"[104] NPT41_PRIV\", \"[105] NPT42_PRIV\", \"[106] NPT43_PRIV\", \"[107] NPT44_PRIV\", \"[108] NPT45_PRIV\", \"[109] PCTPELL\", \"[110] RET_FT4_POOLED_SUPP\", \"[111] RET_FTL4_POOLED_SUPP\", \"[112] RET_PT4_POOLED_SUPP\", \"[113] RET_PTL4_POOLED_SUPP\", \"[114] PCTFLOAN\", \"[115] UG25ABV\", \"[116] MD_EARN_WNE_P10\", \"[117] GT_25K_P6\", \"[118] GT_28K_P6\", \"[119] GRAD_DEBT_MDN_SUPP\", \"[120] GRAD_DEBT_MDN10YR_SUPP\", \"[121] RPY_3YR_RT_SUPP\", \"[122] C150_L4_POOLED_SUPP\", \"[123] C150_4_POOLED_SUPP\", \"[124] UNITID_wikidata\", \"[125] OPEID6_wikidata\", \"[126] STABBR_wikidata\", \"[127] CITY_wikidata\"], \"Datamart ID\": \"D4cb70062-77ed-4097-a486-0b43ffe81463\", \"Recommend Join Columns\": \"INSTNM\", \"Score\": \"0.9050571841886983\", \"URL\": \"http://dsbox02.isi.edu:9000/upload/local_datasets/Most-Recent-Cohorts-Scorecard-Elements.csv\", \"title\": \"most recent cohorts scorecard elements csv\"}, \"supplied_id\": \"DA_college_debt_dataset\", \"supplied_resource_id\": \"learningData\"}"]}}],
                    "inputs": ["denormalize_step"]
                },
                {
                    "name": "to_dataframe_step",
                    "primitives": ["d3m.primitives.data_transformation.dataset_to_dataframe.Common"],
                    "inputs": ["augment_step2"]
                },
                {
                    "name": "common_profiler_step",
                    "primitives": ["d3m.primitives.schema_discovery.profiler.Common"],
                    "inputs": ["to_dataframe_step"]
                },
                {
                  "name": "column_parser_step",
                    "primitives": ["d3m.primitives.data_transformation.column_parser.Common"],
                    "inputs": ["common_profiler_step"]
                },
                {
                    "name": "common_encoder_step",
                    "primitives": ["d3m.primitives.data_transformation.one_hot_encoder.SKlearn"],
                    "inputs": ["column_parser_step"]
                },
                {
                    "name": "extract_attribute_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': (
                                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                                    'https://metadata.datadrivendiscovery.org/types/Attribute',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_encoder_step"]
                },
                {
                    "name": "extract_target_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common",
                        "hyperparameters":
                            {
                                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                                'use_columns': (),
                                'exclude_columns': ()
                            }
                    }],
                    "inputs": ["common_encoder_step"]
                },
                {
                    "name": "model_step",
                    "primitives": [{
                        "primitive": "d3m.primitives.regression.random_forest.SKlearn",
                        "hyperparameters":
                            {
                              'use_semantic_types': [True],
                              'return_result': ['new'],
                              'add_index_columns': [True],
                            }
                    }
                    ],
                    "inputs": ["extract_attribute_step", "extract_target_step"]
                }
                # {
                #   "name": "construct_prediction_step",
                #     "primitives": ["d3m.primitives.data_transformation.construct_predictions.Common"],
                #     "inputs": ["model_step", "common_profiler_step"]
                # }
            ]
        }
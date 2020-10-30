import pandas
import logging
import numpy as np
import d3m.metadata.base as mbase
import d3m.exceptions
from d3m.metadata import hyperparams
from d3m import container
from d3m.primitive_interfaces.featurization import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from . import config


Inputs = container.DataFrame
Outputs = container.List  # [container.DataFrame]


class TimeseriesToListHyperparams(hyperparams.Hyperparams):
    # No parameters required for this primitive
    pass


class TimeseriesToList(TransformerPrimitiveBase[Inputs, Outputs, TimeseriesToListHyperparams]):
    '''
    For timeseries data only:
    Tranform the input Dataframe format data(after DatasetToDataFramePrimitive step)
    into a List data which includes the detail timeseries data of each timeseries csv file
    (Here we would not process the label data and it will be processed separately with other primitives)
    The output may look like:
    -------------------------------------------------------------------------------------
    output_List[0]: The d3mIndex (may not continuous if the dataset was splited previously)
    [12, 20, 40, 6, 22...]
    output_List[1]:
    index     'timeseries_data'
    0       'DataFrame' tpye data
    1       'DataFrame' tpye data
    2       'DataFrame' tpye data
    ...

    -------------------------------------------------------------------------------------
    '''
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-timeseries-to-dataframe',
        'version':config.VERSION,
        'name': "DSBox Timeseries Featurizer dataframe to List Transformer",
        'description': 'Generate a list of DataFrame which includes the details timeseries data',
        'python_path': 'd3m.primitives.data_transformation.time_series_to_list.DSBOX',
        'primitive_family': 'DATA_TRANSFORMATION',
        'algorithm_types': ['DATA_CONVERSION'],
        'keywords': ['timeseries', 'reader'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            'uris': [config.REPOSITORY]
            },
        'installation': [config.INSTALLATION],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: TimeseriesToListHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False
        self.logger = logging.getLogger(__name__)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        elements_amount = inputs.metadata.query((mbase.ALL_ELEMENTS,))['dimension']['length']
        d3mIndex_output = np.asarray(inputs['d3mIndex'].tolist())
        # traverse each selector to check where is the image file
        timeseries_index = -1
        for selector_index in range(elements_amount):
            each_selector = inputs.metadata.query((mbase.ALL_ELEMENTS,selector_index))
            mime_types_found = False
            # here we assume only one column shows the location of the timeseries files
            #print(each_selector)
            if 'mime_types' in each_selector:
                mime_types = (each_selector['mime_types'])
                mime_types_found = True
            elif 'media_types' in each_selector:
                mime_types = (each_selector['media_types'])
                mime_types_found = True
            # do following step only when mime type attriubte found
            if mime_types_found:
                for each_type in mime_types:
                    if ('csv' in each_type):
                        timeseries_index = selector_index
                        location_base_uris = each_selector['location_base_uris'][0]
                        #column_name = each_selector['name']
                        break

        # if no 'csv' related mime_types found, return a empty list
        timeseries_output = []
        if timeseries_index == -1:
            self.logger.info("Warning: Can't find timeseries index!")
            self.logger.info("No timeseries dataset detected!")
            self._has_finished = True
            self._iterations_done = True
            raise d3m.exceptions.InvalidArgumentValueError("no timeseries related metadata found!")

        input_file_name_list = inputs.iloc[:,[timeseries_index]].values.tolist()
        input_file_amount = len(input_file_name_list)
        # create the 4 dimension ndarray for return

        for input_file_number in range(input_file_amount):
            file_path = location_base_uris + input_file_name_list[input_file_number][0]
            file_path = file_path[7:]
            timeseries_output.append(pandas.read_csv(file_path))
            self.logger.info(f"Timeseries data: {file_path} loaded.")
        final_output = container.List([d3mIndex_output, timeseries_output])
        # return a 4-d array (d0 is the amount of the images, d1 and d2 are size of the image, d4 is 3 for color image)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(final_output, self._has_finished)

from d3m.container import ndarray
from d3m.container.pandas import DataFrame
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams
import d3m.metadata.base as mbase
import importlib
import numpy as np
from . import config  # this import not work properly for now

Inputs = DataFrame
Outputs = ndarray
image_size_x = 224
image_size_y = 224
image_layer = 3

class DataFrameToTensorHyperparams(hyperparams.Hyperparams):
    pass

class DataFrameToTensor(TransformerPrimitiveBase[Inputs, Outputs, DataFrameToTensorHyperparams]):
    '''
    For image data only:
    Tranform the input Dataframe format data(after DatasetToDataFramePrimitive step)
    into ndarray format data (finding the image type data and read them)
    so that it could be processed by other image process primitives
    '''
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-dataframe-to-tensor',
        'version':config.VERSION,
        'name': "DSBox Image Featurizer dataframe to tensor transformer",
        'description': 'Generate 4-d ndarray format output from dataframe with images',
        'python_path': 'd3m.primitives.dsbox.DataFrameToTensor',
        'primitive_family': 'DATA_PREPROCESSING',
        'algorithm_types': ["NUMERICAL_METHOD"],
        'keywords': ['image', 'transformer'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            'uris': [ config.REPOSITORY ]
            },
        'installation': [ config.INSTALLATION ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })
    def __init__(self, *, hyperparams: DataFrameToTensorHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._keras_image = importlib.import_module('keras.preprocessing.image')
        self.hyperparams = hyperparams
        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        dataframe_input = inputs
        elements_amount = dataframe_input.metadata.query((mbase.ALL_ELEMENTS,))['dimension']['length']
        # traverse each selector to check where is the image file
        target_index = -1
        for selector_index in range(elements_amount):
            each_selector = inputs.metadata.query((mbase.ALL_ELEMENTS,selector_index))
            mime_types_found = False
            # here we assume only one column shows the location of the target attribute
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
                    if ('image' in each_type):
                        target_index = selector_index
                        location_base_uris = each_selector['location_base_uris'][0]
                        #column_name = each_selector['name']
                        break
        # if no 'image' related mime_types found, return a ndarray with each dimension's length equal to 0
        if (target_index == -1):
            #raise exceptions.InvalidArgumentValueError("no image related metadata found!")
            return CallResult(np.empty(shape=(0,0,0,0)), self._has_finished, self._iterations_done)

        input_file_name_list = inputs.iloc[:,[target_index]].values.tolist()
        input_file_amount = len(input_file_name_list)
        # create the 4 dimension ndarray for return
        tensor_output = np.full((input_file_amount, image_size_x, image_size_y, image_layer), 0)

        for input_file_number in range(input_file_amount):
            file_path = location_base_uris + input_file_name_list[input_file_number][0]
            file_path = file_path[7:]
            im = np.array(self._keras_image.load_img(file_path, target_size=(image_size_x, image_size_y)))
            tensor_output[input_file_number,:,:,:] = im

        # return a 4-d array (d0 is the amount of the images, d1 and d2 are size of the image, d4 is 3 for color image)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(tensor_output, self._has_finished, self._iterations_done)



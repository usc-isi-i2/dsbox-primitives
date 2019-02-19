import importlib
import multiprocessing
import numpy as np
import sys

import d3m.metadata.base as mbase

from d3m.container import List
from d3m.container.pandas import DataFrame
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams

from . import config  # this import not work properly for now

Inputs = DataFrame
Outputs = List
image_size_x = 224
image_size_y = 224
image_layer = 3


class DataFrameToTensorHyperparams(hyperparams.Hyperparams):
    process_amount = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=1,
        description="Specify number of current processes used to read in the images. Default is no multiprocessing.",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )

    resize_X = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=224,
        description="Specify the resized shape[0] of the resized image",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )

    resize_Y = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=224,
        description="Specify the resized shape[1] of the resized image",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )

    image_layer = hyperparams.UniformInt(
        lower=1,
        upper=3,
        upper_inclusive=True,
        default=3,
        description="Specify the output image layer, default value is 3 which corresponds to color images",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )


class DataFrameToTensor(TransformerPrimitiveBase[Inputs, Outputs, DataFrameToTensorHyperparams]):
    '''
    For image data only:
    Tranform the input Dataframe format data(after DatasetToDataFramePrimitive step)
    into ndarray format data (finding the image type data and read them)
    so that it could be processed by other image process primitives

    The output may look like:
    -------------------------------------------------------------------------------------
    output_List[0]:
    The d3mIndex (may not continuous if the dataset was splited previously)
    [12, 20, 40, 6, 22...]

    output_List[1]:
    a 4 dimension numpy nd array:
    dimension 1: The index of each image
    dimension 2: The index of each layer inside the image (R/G/B)
    dimension 3&4: The pixel location of each image of each layer
    each value: each pixel value
    ...

    -------------------------------------------------------------------------------------
    '''
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-dataframe-to-tensor',
        'version':config.VERSION,
        'name': "DSBox Image Featurizer dataframe to tensor transformer",
        'description': 'Generate 4-d ndarray format output from dataframe with images',
        'python_path': 'd3m.primitives.data_preprocessing.DataFrameToTensor.DSBOX',
        'primitive_family': 'DATA_PREPROCESSING',
        'algorithm_types': ["NUMERICAL_METHOD"],
        'keywords': ['image', 'transformer'],
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

    def __init__(self, *, hyperparams: DataFrameToTensorHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._keras_image = importlib.import_module('keras.preprocessing.image')
        self.hyperparams = hyperparams
        self._resize_X = hyperparams['resize_X']
        self._resize_Y = hyperparams['resize_Y']
        self._image_layer = hyperparams['image_layer']
        self._process_amount = hyperparams['process_amount']
        if self._process_amount > multiprocessing.cpu_count():
            self._process_amount = multiprocessing.cpu_count()
        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False

    def _read_one_image(self, file_path: str):
        '''
            Sub sequence that used to read one image
        '''
        im = np.array(self._keras_image.load_img(file_path, target_size=(self._resize_X, self._resize_Y)))
        return im

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        dataframe_input = inputs
        elements_amount = dataframe_input.metadata.query((mbase.ALL_ELEMENTS,))['dimension']['length']
        # traverse each selector to check where is the image file
        target_index = []
        location_base_uris = []

        for selector_index in range(elements_amount):
            each_selector = inputs.metadata.query((mbase.ALL_ELEMENTS, selector_index))
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
                        target_index.append(selector_index)
                        location_base_uris.append(each_selector['location_base_uris'][0])
                        # column_name = each_selector['name']
                        break
        # if no 'image' related mime_types found, return a ndarray with each dimension's length equal to 0
        if (len(target_index) == 0):
            # raise exceptions.InvalidArgumentValueError("no image related metadata found!")
            print("[ERROR] No image related column found!")
            return CallResult(np.empty(shape=(0, 0, 0, 0)), self._has_finished, self._iterations_done)
        elif len(target_index) > 1:
            print("[INFO] Multiple image columns found in the input, this primitive can only handle one column.")

        input_file_name_list = inputs.iloc[:,target_index].values.tolist()
        d3mIndex_output = np.asarray(inputs.index.tolist())

        # generate the list of the file_paths
        location_base_uris = location_base_uris[0]
        if location_base_uris[0:7] == 'file://':
            location_base_uris = location_base_uris[7:]
        # make the file name to be the absolute address
        for i in range(len(input_file_name_list)):
            input_file_name_list[i] = location_base_uris + input_file_name_list[i][0]

        # multi-process part
        try:
            if self._process_amount == 1:
                tensor_output = np.array([self._read_one_image(name) for name in input_file_name_list])
            else:
                with multiprocessing.Pool(self._process_amount) as p:
                    tensor_output = np.array(p.map(self._read_one_image, input_file_name_list))
        # sometimes it may failed with ERROR like "OSError: [Errno 24] Too many open files"
        except OSError as e:
            if e.errno == 24:
                print("Too many open files. Increase limit to 2 * n_trees + 2" +
                      "(unix / mac: ulimit -n [limit], windows: http://bit.ly/2fAKnz0)", file=sys.stderr)
            raise e
        # return a 4-d array (d0 is the amount of the images, d1 and d2 are size of the image, d4 is 3 for color image)
        self._has_finished = True
        self._iterations_done = True
        # final_output[0] is the d3mIndex and final_output[1] is the detail image data
        final_output = [d3mIndex_output, tensor_output]
        return CallResult(final_output, self._has_finished, self._iterations_done)

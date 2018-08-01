""" Feature generation based on deep learning for images
    wrapped into d3m format
    TODO: update primitive info
"""
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m.metadata import hyperparams
from d3m import container

from scipy.misc import imresize

from keras.models import Model
import keras.applications.resnet50 as resnet50
import keras.applications.vgg16 as vgg16

from . import config
#from dsbox.planner.levelone import Primitive

import numpy as np
import sys
import d3m.metadata.base as mbase
# Input image tensor has 4 dimensions: (num_images, 244, 244, 3)
Inputs = container.List

# Output feature has 2 dimensions: (num_images, layer_size[layer_index])
Outputs = container.DataFrame # extracted features

class ResNet50Hyperparams(hyperparams.Hyperparams):
    layer_index = hyperparams.UniformInt(
        lower=0,
        upper=11,
        default=0,
        description="Specify the layer of the neural network to use for features. Lower numbered layers correspond to higher-level abstract features. The number of features by layer index are [2048, 100352, 25088, 25088, 100352, 25088, 25088, 100352, 25088, 25088, 200704].",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    # corresponding layer_size = [2048, 100352, 25088, 25088, 100352, 25088, 25088, 100352, 25088, 25088, 200704]

    generate_metadata = hyperparams.UniformBool(
        default = False,
        description="A control parameter to set whether to generate metada after the feature extraction. It will be very slow if the columns length is very large. For the default condition, it will turn off to accelerate the program running.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )

class Vgg16Hyperparams(hyperparams.Hyperparams):
    layer_index = hyperparams.UniformInt(
        lower=0,
        upper=4,
        default=0,
        description="Specify the layer of the neural network to use for features. Lower numbered layers correspond to higher-level abstract features. The number of features by layer index are [25088, 100352, 200704, 401408]",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    generate_metadata = hyperparams.UniformBool(
        default = False,
        description="A control parameter to set whether to generate metada after the feature extraction. It will be very slow if the columns length is very large. For the default condition, it will turn off to accelerate the program running.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )
    # corresponding layer_size = [25088, 100352, 200704, 401408]

class ResNet50ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, ResNet50Hyperparams]):
    """
    Image Feature Generation using pretrained deep neural network RestNet50.

    Parameters
    ----------
    _layer_index : int, default: 0, domain: range(11)
        Layer of the network to use to generate features. Smaller
        indices are closer to the output layers of the network.

    _resize_data : Boolean, default: True, domain: {True, False}
        If True resize images to 224 by 224.
    """

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-resnet50',
        'version': config.VERSION,
        'name': "DSBox Image Featurizer RestNet50",
        'description': 'Generate image features using RestNet50',
        'python_path': 'd3m.primitives.dsbox.ResNet50ImageFeature',
        'primitive_family': "FEATURE_EXTRACTION",
        'algorithm_types': ["FEEDFORWARD_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'resnet50'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            'uris': [ config.REPOSITORY ]
            },
            # The same path the primitive is registered with entry points in setup.py.
        'installation': [ config.INSTALLATION ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: ResNet50Hyperparams) -> None:

        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False

        #============TODO: these three could be hyperparams=========
        self._layer_index = hyperparams['layer_index']
        self._preprocess_data = True
        self._resize_data = True
        self._RESNET50_MODEL = None
        #===========================================================

        if self._RESNET50_MODEL is None:
            original = sys.stdout
            sys.stdout = sys.stderr
            self._RESNET50_MODEL = resnet50.ResNet50(weights='imagenet')
            sys.stdout = original
        self._layer_numbers = [-2, -4, -8, -11, -14, -18, -21, -24, -30, -33, -36]
        if self._layer_index < 0:
            self._layer_index = 0
        elif self._layer_index > len(self._layer_numbers):
            self._layer_numbers = len(self._layer_numbers)-1

        self._layer_number = self._layer_numbers[self._layer_index]

        self._org_model = self._RESNET50_MODEL
        self._model = Model(self._org_model.input,
                           self._org_model.layers[self._layer_number].output)


        self._annotation = None


    def _preprocess(self, image_tensor):
        """Preprocess image data by modifying it directly"""
        resnet50.preprocess_input(image_tensor)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """Apply neural network-based feature extraction to image_tensor"""
        image_tensor = inputs[1]
        image_d3mIndex = inputs[0]
        # preprocess() modifies the data. For now just copy the data.
        if not len(image_tensor.shape) == 4:
            raise ValueError('Expect shape to have 4 dimension')

        resized = False
        if self._resize_data:
            if not (image_tensor.shape[1] == 244 and image_tensor.shape[2] == 244):
                resized = True
                y = np.empty((image_tensor.shape[0], 224, 224, 3))
                for index in range(image_tensor.shape[0]):
                    y[index] = imresize(image_tensor[index], (224, 224))
                image_tensor = y

        # preprocess() modifies the data. For now just copy the data.
        if self._preprocess_data:
            if resized:
                # Okay to modify image_tensor, since its not input
                data = image_tensor
            else:
                data = image_tensor.copy()
            self._preprocess(data)
        else:
            data = image_tensor
        output_ndarray = self._model.predict(data)
        output_ndarray = output_ndarray.reshape(output_ndarray.shape[0], -1)
        output_dataFrame = container.DataFrame(output_ndarray)
        
        # if generate_metadata is true, update the metadata
        if self.hyperparams["generate_metadata"]:
            for each_column in range(output_ndarray.shape[1]):
                metadata_selector = (mbase.ALL_ELEMENTS,each_column)
                metadata_each_column = {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TabularColumn', 'https://metadata.datadrivendiscovery.org/types/Attribute')}
                output_dataFrame.metadata = output_dataFrame.metadata.update(metadata = metadata_each_column, selector = metadata_selector)
        self._has_finished = True
        self._iterations_done = True
        # update the original index to be d3mIndex
        output_dataFrame = output_dataFrame.set_index(image_d3mIndex)
        return CallResult(output_dataFrame, self._has_finished, self._iterations_done)
'''
    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'ResNet50ImageFeature'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = ''
        self._annotation.ml_algorithm = ['Deep Learning']
        self._annotation.tags = ['feature_extraction' , 'image']
        return self._annotation
'''

class Vgg16ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Vgg16Hyperparams]):
    """
    Image Feature Generation using pretrained deep neural network VGG16.

    Parameters
    ----------
    layer_index : int, default: 0, domain: range(5)
        Layer of the network to use to generate features. Smaller
        indices are closer to the output layers of the network.

    resize_data : Boolean, default: True, domain: {True, False}
        If True resize images to 224 by 224.
    """
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-vgg16',
        'version': config.VERSION,
        'name': "DSBox Image Featurizer VGG16",
        'description': 'Generate image features using VGG16',
        'python_path': 'd3m.primitives.dsbox.Vgg16ImageFeature',
        'primitive_family': "FEATURE_EXTRACTION",
        'algorithm_types': ["FEEDFORWARD_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'vgg16'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            'uris': [ config.REPOSITORY ]
            },
            # The same path the primitive is registered with entry points in setup.py.
        'installation': [ config.INSTALLATION ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })


    def __init__(self, *, hyperparams: Vgg16Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False

        #============TODO: these three could be hyperparams=========
        self._layer_index = self.hyperparams['layer_index']
        self._preprocess_data= True
        self._resize_data = True
        self._VGG16_MODEL = None
        #===========================================================

        if self._VGG16_MODEL is None:
            original = sys.stdout
            sys.stdout = sys.stderr
            self._VGG16_MODEL = vgg16.VGG16(weights='imagenet', include_top=False)
            sys.stdout = original
        self._layer_numbers = [-1, -5, -9, -13, -16]
        if self._layer_index < 0:
            self._layer_index = 0
        elif self._layer_index > len(self._layer_numbers):
            self._layer_index = len(self._layer_numbers)-1

        self._layer_number = self._layer_numbers[self._layer_index]

        self._org_model = self._VGG16_MODEL
        self._model = Model(self._org_model.input,
                           self._org_model.layers[self._layer_number].output)
        self._annotation = None


    def _preprocess(self, image_tensor):
        """Preprocess image data by modifying it directly"""
        vgg16.preprocess_input(image_tensor)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """Apply neural network-based feature extraction to image_tensor"""
        image_tensor = inputs[1]
        image_d3mIndex = inputs[0]

        if not len(image_tensor.shape) == 4:
            raise ValueError('Expect shape to have 4 dimension')

        resized = False
        if self._resize_data:
            if not (image_tensor.shape[1] == 244 and image_tensor.shape[2] == 244):
                resized = True
                y = np.empty((image_tensor.shape[0], 224, 224, 3))
                for index in range(image_tensor.shape[0]):
                    y[index] = imresize(image_tensor[index], (224, 224))
                image_tensor = y

        # preprocess() modifies the data. For now just copy the data.
        if self._preprocess_data:
            if resized:
                # Okay to modify image_tensor, since its not input
                data = image_tensor
            else:
                data = image_tensor.copy()
            self._preprocess(data)
        else:
            data = image_tensor
        output_ndarray = self._model.predict(data)
        output_ndarray = output_ndarray.reshape(output_ndarray.shape[0], -1)
        output_dataFrame = container.DataFrame(container.ndarray(output_ndarray))

        
        # if generate_metadata is true, update the metadata
        if self.hyperparams["generate_metadata"]:
            for each_column in range(output_ndarray.shape[1]):
                metadata_selector = (mbase.ALL_ELEMENTS,each_column)
                metadata_each_column = {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TabularColumn', 'https://metadata.datadrivendiscovery.org/types/Attribute')}
                output_dataFrame.metadata = output_dataFrame.metadata.update(metadata = metadata_each_column, selector = metadata_selector)
        # update the original index to be d3mIndex
        output_dataFrame = output_dataFrame.set_index(image_d3mIndex)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataFrame, self._has_finished, self._iterations_done)

'''
    def annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'Vgg16ImageFeature'
        self._annotation.task = 'FeatureExtraction'
        self._annotation.learning_type = ''
        self._annotation.ml_algorithm = ['Deep Learning']
        self._annotation.tags = ['feature_extraction' , 'image']
        return self._annotation
'''

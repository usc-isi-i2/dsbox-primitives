""" Feature generation based on deep learning for images
    wrapped into d3m format
    TODO: update primitive info
"""
from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from primitive_interfaces.base import CallResult

from d3m_metadata import container, hyperparams, metadata
from d3m_metadata.container import ndarray


from scipy.misc import imresize

from keras.models import Model
import keras.applications.resnet50 as resnet50
import keras.applications.vgg16 as vgg16

from . import config
import typing
#from dsbox.planner.levelone import Primitive

import numpy as np
import sys

Inputs = ndarray  # image tensor, has 4 dimensions
Outputs = ndarray # extracted features


class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """

    pass

class ResNet50ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
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
    metadata = metadata.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-resnet50',
        'version': 'v' + config.VERSION,       
        'name': "DSBox Image Featurizer RestNet50",
        'description': 'Generate image features using RestNet50',
        'python_path': 'd3m.primitives.dsbox.ResNet50ImageFeature',
        'primitive_family': metadata.PrimitiveFamily.FEATURE_EXTRACTION,
        'algorithm_types': [metadata.PrimitiveAlgorithmType.FEEDFORWARD_NEURAL_NETWORK, ],
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

    

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, 
             docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
        # All primitives must define these attributes
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers

        # All other attributes must be private with leading underscore    
        self._has_finished = False
        self._iterations_done = False

        #============TODO: these three could be hyperparams=========
        self._layer_index = 0
        self._preprocess_data=True
        self._resize_data=False
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
        image_tensor = inputs

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
        result = self._model.predict(data)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(result.reshape(result.shape[0], -1), self._has_finished, self._iterations_done)
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

class Vgg16ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
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
    metadata = metadata.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-vgg16',
        'version': 'v' + config.VERSION,       
        'name': "DSBox Image Featurizer VGG16",
        'description': 'Generate image features using VGG16',
        'python_path': 'd3m.primitives.dsbox.Vgg16ImageFeature',
        'primitive_family': metadata.PrimitiveFamily.FEATURE_EXTRACTION,
        'algorithm_types': [metadata.PrimitiveAlgorithmType.FEEDFORWARD_NEURAL_NETWORK, ],
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


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, 
             docker_containers: typing.Union[typing.Dict[str, str], None] = None) -> None:
        # All primitives must define these attributes
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers

        # All other attributes must be private with leading underscore    
        self._has_finished = False
        self._iterations_done = False

        #============TODO: these three could be hyperparams=========
        self._layer_index = 0
        self._preprocess_data= True
        self._resize_data = False
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
        image_tensor = inputs

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
        result = self._model.predict(data)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(result.reshape(result.shape[0], -1), self._has_finished, self._iterations_done)

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

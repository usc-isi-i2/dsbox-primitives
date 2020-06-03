""" Feature generation based on deep learning for images
    wrapped into d3m format
    TODO: update primitive info
"""

import importlib
import logging
import os
import sys
import typing

import numpy as np
import pandas as pd

from skimage.transform import resize as imresize

from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams
from d3m import container

from .utils import image_utils
from . import config

# If True use tensorflow.keras.* modules, instead of keras.*
KERAS_MODULE = 'keras'
TENSORFLOW_MODULE = 'tensorflow.keras'
USE_MODULE = TENSORFLOW_MODULE

# Input image tensor has 4 dimensions: (num_images, 224, 224, 3)
Inputs = container.List
Inputs_inceptionV3 = container.DataFrame
# Output feature has 2 dimensions: (num_images, layer_size[layer_index])
Outputs = container.DataFrame  # extracted features


class InceptionV3Hyperparams(hyperparams.Hyperparams):
    minimum_frame = hyperparams.UniformInt(
        lower=1,
        upper=100000,
        default=40,
        description="Specify the least amount of the frame in a video, if the video is too short with less frame, "
                    "we will not consider to use for training",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    maximum_frame = hyperparams.UniformInt(
        lower=1,
        upper=100000,
        default=300,
        description="Specify the max amount of the frame in a video, if the video is too long, "
                    "we will not consider to use for training",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    generate_metadata = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to generate metada after the feature extraction."
                    " It will be very slow if the columns length is very large. For the default condition, "
                    "it will turn off to accelerate the program running.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    do_preprocess = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to do preprocess step on input tensor, it normally should be set as true",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )

    do_resize = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to resize the input tensor to be correct shape as input, "
                    "it normally should be set as true",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    use_limitation = hyperparams.UniformBool(
        default=True,
        description="A control parameter to consider the limitation of the maximum/minimum frame amount during processing. "
                    "If set False, we will ignore the input videos outsite the frame amount limitation.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )


class ResNet50Hyperparams(hyperparams.Hyperparams):
    layer_index = hyperparams.UniformInt(
        lower=0,
        upper=11,
        default=0,
        description="Specify the layer of the neural network to use for features. "
                    "Lower numbered layers correspond to higher-level abstract features. "
                    "The number of features by layer index are "
                    "[2048, 100352, 25088, 25088, 100352, 25088, 25088, 100352, 25088, 25088, 200704].",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    # corresponding layer_size = [2048, 100352, 25088, 25088, 100352, 25088, 25088, 100352, 25088, 25088, 200704]

    generate_metadata = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to generate metada after the feature extraction. "
                    "It will be very slow if the columns length is very large. For the default condition, "
                    "it will turn off to accelerate the program running.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )


class Vgg16Hyperparams(hyperparams.Hyperparams):
    layer_index = hyperparams.UniformInt(
        lower=0,
        upper=4,
        default=0,
        description="Specify the layer of the neural network to use for features. "
                    "Lower numbered layers correspond to higher-level abstract features. "
                    "The number of features by layer index are [25088, 100352, 200704, 401408]",
        semantic_types=["http://schema.org/Integer", "https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )

    generate_metadata = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to generate metada after the feature extraction. "
                    "It will be very slow if the columns length is very large. For the default condition, "
                    "it will turn off to accelerate the program running.",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    # corresponding layer_size = [25088, 100352, 200704, 401408]


class KerasPrimitive:
    _weight_files = []

    def __init__(self):
        self._initialized = False
        self.logger = logging.getLogger(__name__)

    def _lazy_init(self):
        if self._initialized:
            return

        # Lazy import modules as not to slow down d3m.index
        global keras_models, keras_backend, tf
        if USE_MODULE == TENSORFLOW_MODULE:
            self.logger.info('Using tensorflow.keras modules')
            keras_models = importlib.import_module('tensorflow.keras.models')
            keras_backend = importlib.import_module('tensorflow.keras.backend')
        elif USE_MODULE == KERAS_MODULE:
            self.logger.info('Using keras modules')
            keras_models = importlib.import_module('keras.models')
            keras_backend = importlib.import_module('keras.backend')
        tf = importlib.import_module('tensorflow')

        self._initialized = True

    @staticmethod
    def _get_keras_data_dir(cache_subdir='models'):
        """
        Return Keras cache directory. See keras/utils/data_utils.py:get_file()
        """
        cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
        datadir_base = os.path.expanduser(cache_dir)
        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', '.keras')
        datadir = os.path.join(datadir_base, cache_subdir)
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        return datadir

    @staticmethod
    def _get_weight_installation(weight_files: typing.List['WeightFile']):
        """
        Return D3M file installation entries
        """
        return [
            {'type': 'FILE',
             'key': weight_file.name,
             'file_uri': weight_file.uri,
             'file_digest': weight_file.digest}
            for weight_file in weight_files]

    def _setup_weight_files(self) -> str:
        """
        return the weight file location for model to load
        """
        for file_info in self._weight_files:
            if USE_MODULE not in file_info.package:
                continue
            if file_info.name in self.volumes:
                return self.volumes[file_info.name]
                # dest = os.path.join(file_info.data_dir, file_info.name)
                # if not os.path.exists(dest):
                # shutil.copy2(self.volumes[file_info.name], dest)
            else:
                # can only try to let keras load by itself
                self.logger.warning('Keras weight file not in volume: {}'.format(file_info.name))
                return "imagenet"

    def _get_weight_file(self, name):
        file_info = [info for info in self._weight_files if info.name == name and USE_MODULE in info.package]
        if not file_info:
            print(self._weight_files)
            raise ValueError(f'Cannot find weight file definition: {name}')
        if len(file_info) > 1:
            self.logger.warning('Found mulitple weight file definitions: %s', file_info)
        file_info = file_info[0]
        if file_info.name in self.volumes:
            return self.volumes[file_info.name]
        return ""


class WeightFile(typing.NamedTuple):
    name: str
    uri: str
    digest: str
    package: str = [KERAS_MODULE, TENSORFLOW_MODULE]
    data_dir: str = KerasPrimitive._get_keras_data_dir()


class ResNet50ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, ResNet50Hyperparams], KerasPrimitive):
    """
    Image Feature Generation using pretrained deep neural network ResNet50

    Parameters
    ----------
    _layer_index : int, default: 0, domain: range(11)
        Layer of the network to use to generate features. Smaller
        indices are closer to the output layers of the network.

    _resize_data : Boolean, default: True, domain: {True, False}
        If True resize images to 299x299.
    """

    # Resnet50 weight files info is from here:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
    _weight_files = [
        WeightFile(KERAS_MODULE + ':resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                   ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.2/'
                    'resnet50_weights_tf_dim_ordering_tf_kernels.h5'),
                   'bdc6c9f787f9f51dffd50d895f86e469cc0eb8ba95fd61f0801b1a264acb4819',
                   'keras'),
        WeightFile(TENSORFLOW_MODULE + ':resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                   'https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                   '7011d39ea4f61f4ddb8da99c4addf3fae4209bfda7828adb4698b16283258fbe',
                   'tensorflow.keras'),
        # WeightFile('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #            ('https://github.com/fchollet/deep-learning-models/'
        #             'releases/download/v0.2/'
        #             'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'),
        #            'a268eb855778b3df3c7506639542a6af')
    ]
    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-resnet50',
        'version': config.VERSION,
        'name': "DSBox Image Featurizer ResNet50",
        'description': 'Generate image features using ResNet50',
        'python_path': 'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX',
        'primitive_family': "FEATURE_EXTRACTION",
        'algorithm_types': ["FEEDFORWARD_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'resnet50'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            'uris': [config.REPOSITORY]
        },
        # The same path the primitive is registered with entry points in setup.py.
        'installation': [config.INSTALLATION] + KerasPrimitive._get_weight_installation(_weight_files),
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: ResNet50Hyperparams, volumes: typing.Union[typing.Dict[str, str], None] = None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        KerasPrimitive.__init__(self)
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False
        # ============TODO: these three could be hyperparams=========
        self._layer_index = hyperparams['layer_index']
        self._preprocess_data = True
        self._resize_data = True
        self.logger = logging.getLogger(__name__)
        # ===========================================================

    def _lazy_init(self):
        if self._initialized:
            return

        KerasPrimitive._lazy_init(self)

        # Lazy import modules as not to slow down d3m.index
        global resnet50
        if USE_MODULE == TENSORFLOW_MODULE:
            resnet50 = importlib.import_module('tensorflow.keras.applications.resnet50')
        else:
            resnet50 = importlib.import_module('keras.applications.resnet50')

        # self._setup_weight_files()
        weight_file = self._get_weight_file(USE_MODULE + ':resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        if not weight_file:
            weight_file = 'imagenet'
        self.logger.debug('Using weight file: %s', weight_file)

        keras_backend.clear_session()

        # Always create a new model
        original = sys.stdout
        sys.stdout = sys.stderr
        self._org_model = resnet50.ResNet50(weights=weight_file)
        sys.stdout = original

        self._layer_numbers = [-2, -4, -8, -11, -14, -18, -21, -24, -30, -33, -36]
        if self._layer_index < 0:
            self._layer_index = 0
        elif self._layer_index > len(self._layer_numbers):
            self._layer_numbers = len(self._layer_numbers) - 1

        self._layer_number = self._layer_numbers[self._layer_index]

        self._model = keras_models.Model(self._org_model.input,
                                         self._org_model.layers[self._layer_number].output)

        # self._graph = tf.get_default_graph()
        # self._graph = tf.compat.v1.get_default_graph()

        self._annotation = None

    def _preprocess(self, image_tensor):
        """Preprocess image data by modifying it directly"""
        return resnet50.preprocess_input(image_tensor)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """Apply neural network-based feature extraction to image_tensor"""

        self._lazy_init()

        image_tensor = inputs[1]
        image_d3mIndex = inputs[0]
        # preprocess() modifies the data. For now just copy the data.
        if not len(image_tensor.shape) == 4:
            raise ValueError('Expect shape to have 4 dimension')

        resized = False
        if self._resize_data:
            if not (image_tensor.shape[1] == 224 and image_tensor.shape[2] == 224):
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
            data = self._preprocess(data)
        else:
            data = image_tensor

        # # BUG fix: add global variable to fix ta3 system if calling multiple times of this primitive
        # with self._graph.as_default():
        #    output_ndarray = self._model.predict(data)
        output_ndarray = self._model.predict(data)

        output_ndarray = output_ndarray.reshape(output_ndarray.shape[0], -1).astype(
            'float64')  # change to astype float64 to resolve pca output

        output_dataFrame = container.DataFrame(output_ndarray)

        # update the original index to be d3mIndext
        output_dataFrame = pd.concat([pd.DataFrame(image_d3mIndex, columns=['d3mIndex']), pd.DataFrame(output_dataFrame)], axis=1)
        output_dataFrame = container.DataFrame(output_dataFrame, generate_metadata=False)

        if self.hyperparams["generate_metadata"]:
            output_dataFrame = image_utils.generate_all_metadata(output_dataFrame)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataFrame, self._has_finished)


class Vgg16ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Vgg16Hyperparams], KerasPrimitive):
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
    # vgg16 weight files info is from here:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
    _weight_files = [
        # WeightFile(
        #     'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        #     ('https://github.com/fchollet/deep-learning-models/'
        #      'releases/download/v0.1/'
        #      'vgg16_weights_tf_dim_ordering_tf_kernels.h5'),
        #     '64373286793e3c8b2b4e3219cbf3544b'
        # ),
        WeightFile(
            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            ('https://github.com/fchollet/deep-learning-models/'
             'releases/download/v0.1/'
             'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'),
            'bfe5187d0a272bed55ba430631598124cff8e880b98d38c9e56c8d66032abdc1'
        )
    ]
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-vgg16',
        'version': config.VERSION,
        'name': "DSBox Image Featurizer VGG16",
        'description': 'Generate image features using VGG16',
        'python_path': 'd3m.primitives.feature_extraction.vgg16_image_feature.DSBOX',
        'primitive_family': "FEATURE_EXTRACTION",
        'algorithm_types': ["FEEDFORWARD_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'vgg16'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            'uris': [config.REPOSITORY]
        },
        # The same path the primitive is registered with entry points in setup.py.
        'installation': [config.INSTALLATION] + KerasPrimitive._get_weight_installation(_weight_files),
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: Vgg16Hyperparams, volumes: typing.Union[typing.Dict[str, str], None] = None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        KerasPrimitive.__init__(self)
        self.hyperparams = hyperparams

        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False
        # ============TODO: these three could be hyperparams=========
        self._layer_index = self.hyperparams['layer_index']
        self._preprocess_data = True
        self._resize_data = True
        # ===========================================================

    def _lazy_init(self):
        if self._initialized:
            return

        KerasPrimitive._lazy_init(self)

        # Lazy import modules as not to slow down d3m.index
        global vgg16
        if USE_MODULE == TENSORFLOW_MODULE:
            vgg16 = importlib.import_module('tensorflow.keras.applications.vgg16')
        else:
            vgg16 = importlib.import_module('keras.applications.vgg16')

        weights_loc = self._setup_weight_files()

        keras_backend.clear_session()

        original = sys.stdout
        sys.stdout = sys.stderr
        self._org_model = vgg16.VGG16(weights=weights_loc, include_top=False)
        sys.stdout = original

        self._layer_numbers = [-1, -5, -9, -13, -16]
        if self._layer_index < 0:
            self._layer_index = 0
        elif self._layer_index > len(self._layer_numbers):
            self._layer_index = len(self._layer_numbers) - 1

        self._layer_number = self._layer_numbers[self._layer_index]

        self._model = keras_models.Model(self._org_model.input,
                                         self._org_model.layers[self._layer_number].output)

        # self._graph = tf.get_default_graph()
        self._graph = tf.compat.v1.get_default_graph()

        self._annotation = None

    def _preprocess(self, image_tensor):
        """Preprocess image data by modifying it directly"""
        return vgg16.preprocess_input(image_tensor)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """Apply neural network-based feature extraction to image_tensor"""

        self._lazy_init()

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
            data = self._preprocess(data)
        else:
            data = image_tensor

        # BUG fix: add global variable to fix ta3 system if calling multiple times of this primitive
        # with self._graph.as_default():
        #     output_ndarray = self._model.predict(data)

        output_ndarray = self._model.predict(data)
        output_ndarray = output_ndarray.reshape(output_ndarray.shape[0], -1).astype(
            'float64')  # change to astype float64 to resolve pca output
        output_dataFrame = container.DataFrame(output_ndarray)

        # update the original index to be d3mIndex
        output_dataFrame = pd.concat([pd.DataFrame(image_d3mIndex, columns=['d3mIndex']), pd.DataFrame(output_dataFrame)], axis=1)
        output_dataFrame = container.DataFrame(output_dataFrame, generate_metadata=False)

        # add other metadata
        if self.hyperparams["generate_metadata"]:
            output_dataFrame = image_utils.generate_all_metadata(output_dataFrame)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataFrame, self._has_finished)


class InceptionV3ImageFeature(FeaturizationTransformerPrimitiveBase[Inputs_inceptionV3, Outputs, InceptionV3Hyperparams],
                              KerasPrimitive):
    """
    Image Feature Generation using pretrained deep neural network inception_V3.
    """

    # inception_V3 weight files info is from here:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
    _weight_files = [
        WeightFile('inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                   ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.5/'
                    'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'),
                   '00c9ea4e4762f716ac4d300d6d9c2935639cc5e4d139b5790d765dcbeea539d0'),
    ]

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-image-inceptionV3',
        'version': config.VERSION,
        'name': "DSBox Image Featurizer inceptionV3",
        'description': 'Generate image features using InceptionV3 model',
        'python_path': 'd3m.primitives.feature_extraction.inceptionV3_image_feature.DSBOX',
        'primitive_family': "FEATURE_EXTRACTION",
        'algorithm_types': ["FEEDFORWARD_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'inceptionV3'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            'uris': [config.REPOSITORY]
        },
        # The same path the primitive is registered with entry points in setup.py.
        'installation': [config.INSTALLATION] + KerasPrimitive._get_weight_installation(_weight_files),
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: InceptionV3Hyperparams, volumes: typing.Union[typing.Dict[str, str], None] = None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        KerasPrimitive.__init__(self)
        self.hyperparams = hyperparams
        self._minimum_frame = self.hyperparams['minimum_frame']
        self._maximum_frame = self.hyperparams['maximum_frame']
        self._preprocess_data = self.hyperparams['do_preprocess']
        self._resize_data = self.hyperparams['do_resize']
        self._use_limitation = self.hyperparams['use_limitation']
        # All other attributes must be private with leading underscore
        self._has_finished = False
        self._iterations_done = False
        self._model = None
        # ===========================================================

    def _lazy_init(self):
        if self._initialized:
            return

        KerasPrimitive._lazy_init(self)

        # Lazy import modules as not to slow down d3m.index
        global inception_v3
        if USE_MODULE == TENSORFLOW_MODULE:
            inception_v3 = importlib.import_module('tensorflow.keras.applications.inception_v3')
        else:
            inception_v3 = importlib.import_module('keras.applications.inception_v3')

        weights_loc = self._setup_weight_files()

        keras_backend.clear_session()

        self._model = None
        if self._model is None:
            original = sys.stdout
            sys.stdout = sys.stderr
            self._model = inception_v3.InceptionV3(weights=weights_loc, include_top=True)
            sys.stdout = original

        self._org_model = self._model
        self._model = keras_models.Model(
            inputs=self._org_model.input,
            outputs=self._org_model.get_layer('avg_pool').output
        )
        # self._graph = tf.get_default_graph()
        # self._graph = tf.compat.v1.get_default_graph()

        self._annotation = None

    def produce(self, *, inputs: Inputs_inceptionV3, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """Apply neural network-based feature extraction to image_tensor"""

        self._lazy_init()
        # take a look at the first row's last columns which should come from common-primitive's video reader
        # if it is a ndarray type data and it has 4 dimension, we can be sure that input has video format data
        if len(inputs.index) > 0 and type(inputs.iloc[0, -1]) is container.ndarray and len(inputs.iloc[0, -1].shape) == 4:
            # send the video ndarray part only
            # TODO: we now use fixed columns (last column), it should be updated to use semantic types to check
            extracted_feature_dataframe = self._produce(inputs.iloc[:, -1])
        else:
            raise ValueError('No video format input found from inputs.')

        # combine the extracted_feature_dataframe and input daraframe (but remove the video tensor to reduce the size of dataframe)
        last_column = len(inputs.columns) - 1
        output_dataFrame = inputs.iloc[:, :-1]
        output_dataFrame = output_dataFrame.reset_index()
        output_dataFrame['extraced_features'] = extracted_feature_dataframe
        output_dataFrame = image_utils.generate_all_metadata(output_dataFrame)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataFrame, self._has_finished)

    def _preprocess(self, image_tensor):
        """Preprocess image data"""
        return inception_v3.preprocess_input(image_tensor)

    def _produce(self, input_videos):
        """
            Inner produce function, will return a dataframe with the extraed features column only
        """
        features = []
        # extracted_feature_dataframe = container.DataFrame(columns = ["extraced_features"])
        for i, each_video in enumerate(input_videos):
            self.logger.info("Now processing No. " + str(i) + " video.")
            frame_number = each_video.shape[0]
            if self._use_limitation and (frame_number > self._maximum_frame or frame_number < self._minimum_frame):
                self.logger.info("skip No. %s", str(i))
                features.append(None)
                continue
            each_feature = self._process_one_video(each_video)
            features.append(each_feature)
            extracted_feature_dataframe = container.DataFrame({"extraced_features": features}, generate_metadata=False)

        return extracted_feature_dataframe

    def _process_one_video(self, input_video_ndarray):
        """
            Inner function to process the input video ndarry type data
            and output a ndarry with extracted features
        """
        # initialize
        channel_at_first = False
        # check input video's frame format
        frame_number = input_video_ndarray.shape[0]
        if len(input_video_ndarray[0].shape) == 3:
            channel_number = 3
            if input_video_ndarray[0].shape[0] == 3:
                channel_at_first = True
        else:
            channel_number = 1

        # start processing
        processed_input = np.empty((self._minimum_frame, 299, 299, channel_number))
        if frame_number < self._minimum_frame:
            skip = frame_number / float(self._minimum_frame)
        else:
            skip = frame_number // self._minimum_frame

        for count in range(self._minimum_frame):
            origin_frame_number = round(count * skip)
            each_frame = input_video_ndarray[origin_frame_number]
            if channel_number == 3 and channel_at_first:
                each_frame = np.moveaxis(each_frame, 0, 2)
            elif channel_number == 1:
                # if it is gray image(with only one channel)
                each_frame = np.expand_dims(each_frame, axis=2)
            if self._resize_data:
                each_frame = imresize(each_frame, (299, 299))
            # put the processed frame into new output
            processed_input[count] = each_frame
        # run preprocess step
        processed_input = self._preprocess(processed_input)

        # processed_input_tensor = tf.constant(processed_input, dtype = tf.float32)
        features = self._model.predict(processed_input)
        return features

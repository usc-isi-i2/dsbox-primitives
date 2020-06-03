import logging
import numpy as np
import os
import importlib
import sys
import typing
import cv2
import copy
import pandas as pd
import tensorflow as tf
import collections
from datetime import datetime
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import container
from .utils import image_utils
from . import config

Inputs = container.DataFrame
Outputs = container.DataFrame


class YoloHyperparams(hyperparams.Hyperparams):
    use_fitted_weight = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to use the pre-trained model weights or train new model",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    output_to_tmp_dir = hyperparams.UniformBool(
        default=False,
        description="whether to output the images with bounding boxes and retrained model for debugging purpose",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    data_augmentation = hyperparams.UniformBool(
        default=True,
        description="whether to run some augmentation step on training input images,"
                    " not valid if use_fitted_weight is set to True",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    epochs = hyperparams.UniformInt(
        default=50,
        lower=1,
        upper=sys.maxsize,
        description="The epochs aimed to run on tuning, not valid if use_fitted_weight is set to True",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    warmup_epochs = hyperparams.UniformInt(
        default=2,
        lower=1,
        upper=sys.maxsize,
        description="The warmup epochs aimed to run on tuning, not valid if use_fitted_weight is set to True",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    train_batch_size = hyperparams.UniformInt(
        default=4,
        lower=1,
        upper=sys.maxsize,
        description="The batch size for each training step, not valid if use_fitted_weight is set to True",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    test_batch_size = hyperparams.UniformInt(
        default=4,
        lower=1,
        upper=sys.maxsize,
        description="The batch size for each test step, not valid if use_fitted_weight is set to True",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    blob_scale_factor = hyperparams.Uniform(
        default=0.00392,
        lower=0,
        upper=1,
        description="multiplier for image values for cv.dnn.blobFromImage function",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    blob_output_shape_x = hyperparams.UniformInt(
        default=416,
        lower=0,
        upper=sys.maxsize,
        description=" spatial size for output image (x-dimension) in blob",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    blob_output_shape_y = hyperparams.UniformInt(
        default=416,
        lower=0,
        upper=sys.maxsize,
        description=" spatial size for output image (y-dimension) in blob",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    lr_init = hyperparams.Uniform(
        default=1e-3,
        lower=0,
        upper=1,
        description="initial loss rate setting, not valid if use_fitted_weight is set to True",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    lr_end = hyperparams.Uniform(
        default=1e-6,
        lower=0,
        upper=1,
        description="end loss rate setting, not valid if use_fitted_weight is set to True",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"]
    )
    confidences_threshold = hyperparams.Uniform(
        default=0.5,
        lower=0,
        upper=1,
        description="threshold of the confident to use the predictions",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )
    nms_threshold = hyperparams.Uniform(
        default=0.4,
        lower=0,
        upper=1,
        description="threshold of the non-max suppression",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
    )


class Params(params.Params):
    target_class_id: typing.List[int]
    output_layer: typing.List[str]
    target_column_name: str
    input_image_column_name: str
    dump_model_path: typing.Optional[str]


class Yolo(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, YoloHyperparams]):
    """
    Object detection primitive that use YOLOv3 algorithm
    offical site available at : https://pjreddie.com/darknet/yolo/

    Parameters
    ----------
    target_class_id: The id of the object detection targets
    output_layer: The list of the output layer number from YOLO's DNN
    """
    _weight_files = [{'type': 'FILE',
                      'key': 'yolov3.weights',
                      'file_uri': "https://pjreddie.com/media/files/yolov3.weights",
                      'file_digest': "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"}]

    __author__ = 'USC ISI'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': 'dsbox-featurizer-object-detection-yolo',
        'version': config.VERSION,
        'name': "DSBox Object Detection YOLO",
        'description': 'Find the corresponding object position from given images(tensors)',
        'python_path': 'd3m.primitives.feature_extraction.yolo.DSBOX',
        'primitive_family': "FEATURE_EXTRACTION",
        'algorithm_types': ["DEEP_NEURAL_NETWORK"],
        'keywords': ['image', 'featurization', 'yolo'],
        'source': {
            'name': config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            'uris': [config.REPOSITORY]
        },
        # The same path the primitive is registered with entry points in setup.py.
        'installation': [config.INSTALLATION] + _weight_files,
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.

        # A metafeature about preconditions required for this primitive to operate well.
        'precondition': [],
        'hyperparms_to_tune': []
    })

    def __init__(self, *, hyperparams: YoloHyperparams, volumes: typing.Union[typing.Dict[str, str], None] = None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        self.hyperparams = hyperparams
        # All other attributes must be private with leading underscore
        self._fitted = False
        self._has_finished = False
        self._iterations_done = False
        self._inited = False
        self._model = None
        self._training_inputs = None
        self._training_outputs = None
        self._object_names = None
        self._target_column_name: str = ""
        self._target_class_id: typing.List[int] = []
        self._location_base_uris = ""
        self._output_layer: typing.List[str] = []

        self.optimizer = None
        self._loaded_dataset = None
        self.global_steps = None
        self.warmup_steps = None
        self.total_steps = None
        self._loaded_dataset = None
        self._dump_model_path = ""
        self._current_phase = None
        self.logger = logging.getLogger(__name__)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
            If using the pre-training model, here we will use this model to detect what inside the bounding boxes from
            training dataset. Then, count the number of the objects detected in each box, we will treat only the objects
            amount number larger than the threshold to be the target that we need to detect in the test part.
        """
        if self._fitted:
            self.logger.error("The model has already been fitted once! Should not fit again.")
            return CallResult(None)

        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        # initialization
        self._lazy_init("train")

        if self.hyperparams['use_fitted_weight']:
            self.fit_on_pretrained_model()
        else:
            self.retrain()

        self._fitted = True
        self._has_finished = True
        return CallResult(None, has_finished=self._has_finished, iterations_done=1)

    def fit_on_pretrained_model(self) -> None:
        conf_threshold = self.hyperparams['confidences_threshold']
        class_count = [0 for i in range(len(self._object_names))]
        memo = {}
        for i, each_image_name in enumerate(self._training_inputs[self._input_image_column_name]):
            if each_image_name in memo:
                each_image = memo[each_image_name]
            else:
                each_image = cv2.imread(os.path.join(self._location_base_uris, each_image_name))
                memo[each_image_name] = each_image

            ground_truth_box = self._training_outputs[self._target_column_name].iloc[i]
            if isinstance(ground_truth_box, str):
                ground_truth_box = ground_truth_box.split(",")
            self.logger.debug("processing {} on {}".format(each_image_name, str(ground_truth_box)))
            # update 2019.5.9: cut the image into the bounding box area only
            each_image_cutted = self._cut_image(each_image, ground_truth_box)
            # Creates 4-dimensional blob from image.
            # swapRB has to be True, otherwise the channel is not R,G,B style
            blob = cv2.dnn.blobFromImage(each_image_cutted,
                                         self.hyperparams["blob_scale_factor"],
                                         (self.hyperparams["blob_output_shape_x"], self.hyperparams["blob_output_shape_y"]),
                                         (0, 0, 0), True, crop=False)
            # set input blob for the network
            self._model.setInput(blob)
            outs = self._model.forward(self._output_layer)

            # initialization for each image
            class_ids = []
            confidences = []
            boxes = []
            width = each_image_cutted.shape[1]
            height = each_image_cutted.shape[0]
            # for each detection from each output layer
            # get the confidence, class id, bounding box params
            # and ignore weak detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # apply non-max suppression to combine duplicate bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, self.hyperparams["nms_threshold"])

            if len(indices) > 0:
                for each in indices:
                    i = each[0]
                    box = boxes[i]
                    class_count[class_ids[i]] += 1
            else:
                self.logger.warning("No object detected on {} on [{}]".format(each_image_name, str(ground_truth_box)))

        # find the real target that we need to detect
        for i, each in enumerate(class_count):
            if each >= conf_threshold * len(memo):
                self._target_class_id.append(i)
        if len(self._target_class_id) < 1:
            self.logger.error("No corresponding target object detected in training set with pre-trained model")
        elif len(self._target_class_id) > 1:
            self.logger.warning("More than 1 target object detected in the training set with pre-trained model")

        target_id_str = ",".join(self._object_names[x] for x in self._target_class_id)
        self.logger.info("The target class id is: [" + target_id_str + "]")

    def get_params(self) -> Params:
        param = Params(
            target_class_id=self._target_class_id,
            output_layer=self._output_layer,
            target_column_name=self._target_column_name,
            input_image_column_name=self._input_image_column_name,
            dump_model_path=self._dump_model_path
        )
        return param

    def set_params(self, *, params: Params) -> None:
        self._target_class_id = params["target_class_id"]
        self._output_layer = params["output_layer"]
        self._target_column_name = params["target_column_name"]
        self._input_image_column_name = params["input_image_column_name"]
        self._dump_model_path = params["dump_model_path"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs

        target_column_names = []
        for i in range(outputs.shape[1]):
            each_selector = (metadata_base.ALL_ELEMENTS, i)
            each_column_meta = outputs.metadata.query(each_selector)
            if "https://metadata.datadrivendiscovery.org/types/TrueTarget" in each_column_meta["semantic_types"]:
                target_column_names.append(outputs.columns[i])

        if len(target_column_names) > 1:
            self.logger.warning("Multiple target detected!")
        self._target_column_name = target_column_names[0]

        input_column_names = []
        for i in range(inputs.shape[1]):
            each_selector = (metadata_base.ALL_ELEMENTS, i)
            each_column_meta = inputs.metadata.query(each_selector)
            if "http://schema.org/ImageObject" in each_column_meta["semantic_types"]:
                input_column_names.append(inputs.columns[i])

        if len(input_column_names) > 1:
            self.logger.warning("Multiple input image columns detected!")
        self._input_image_column_name = input_column_names[0]

        self._location_base_uris = image_utils.get_image_path(inputs)
        self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Use YOLO to detect the objects in the input dataframe
            The function will read the images if the input is a dataframe with image names
            The detected output depends on the input training dataset's ojbects
        """
        input_copy = copy.copy(inputs)
        scale = self.hyperparams['blob_scale_factor']
        blob_x = self.hyperparams['blob_output_shape_x']
        blob_y = self.hyperparams['blob_output_shape_y']
        self._lazy_init("test")
        self._location_base_uris = image_utils.get_image_path(input_copy)

        if self.hyperparams["use_fitted_weight"]:
            output_dataframe = self._produce_for_fitted_weight(input_copy)
        else:
            output_dataframe = self._produce_for_retrain_weights(input_copy)

        output_dataframe = container.DataFrame(output_dataframe, generate_metadata=False)

        # add metadata
        metadata_selector = (metadata_base.ALL_ELEMENTS, 0)
        output_dataframe.metadata = output_dataframe.metadata.update(metadata=input_copy.metadata.query(metadata_selector),
                                                                     selector=metadata_selector)

        # add prediction column's metadata
        for each_column in range(1, output_dataframe.shape[1]):
            metadata_selector = (metadata_base.ALL_ELEMENTS, each_column)
            metadata_each_column = {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PredictedTarget',)}
            output_dataframe.metadata = output_dataframe.metadata.update(metadata=metadata_each_column,
                                                                         selector=metadata_selector)

        # add shape metadata
        metadata_shape_part_dict = image_utils.generate_metadata_shape_part(value=output_dataframe, selector=())
        for each_selector, each_metadata in metadata_shape_part_dict.items():
            output_dataframe.metadata = output_dataframe.metadata.update(selector=each_selector, metadata=each_metadata)
        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataframe, self._has_finished, self._iterations_done)

    def _produce_for_fitted_weight(self, input_df):
        """
            produce function for using the pretrained model weights
        """
        bbox_count = 0
        memo = set()
        output_df_dict = {}
        for i, each_row in input_df.iterrows():
            each_image_name = each_row[self._input_image_column_name]
            if each_image_name in memo:
                continue
            else:
                memo.add(each_image_name)
            self.logger.debug("Predicting on {}".format(each_image_name))
            image_path = os.path.join(self._location_base_uris, each_image_name)
            each_image = cv2.imread(image_path)
            if each_image is None:
                self.logger.error("loading image from {} failed!".format(str(image_path)))
                continue
            self.logger.debug("Now detecting objects in {}".format(each_image_name))
            # Creates 4-dimensional blob from image.
            # swapRB has to be True, otherwise the channel is not R,G,B style
            blob = cv2.dnn.blobFromImage(each_image, self.hyperparams["blob_scale_factor"],
                                         (self.hyperparams["blob_output_shape_x"], self.hyperparams["blob_output_shape_y"]),
                                         (0, 0, 0), True, crop=False
                                         )
            # set input blob for the network
            self._model.setInput(blob)
            # run inference through the network
            # and gather predictions from output layers
            outs = self._model.forward(self._output_layer)

            # initialization
            class_ids = []
            confidences = []
            boxes = []
            width = each_image.shape[1]
            height = each_image.shape[0]
            # for each detetion from each output layer
            # get the confidence, class id, bounding box params
            # and ignore weak detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.hyperparams["confidences_threshold"]:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # apply non-max suppression to combine duplicate bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.hyperparams["confidences_threshold"],
                                       self.hyperparams["nms_threshold"])
            if len(indices) > 0:
                for each in indices:
                    i = each[0]
                    if class_ids[i] in self._target_class_id:
                        box = boxes[i]
                        bbox_count += 1
                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2])
                        h = int(box[3])
                        box_result = [str(x), str(y), str(x), str(y + h), str(x + w), str(y + h), str(x + w), str(y)]
                        box_result = ",".join(box_result)  # remove "[" and "]"
                        # box_result = str(x)+","+str(y) + "," +str(x+w)+ ","+str(y+h)
                        output_df_dict[bbox_count] = {"d3mIndex": each_row["d3mIndex"], self._target_column_name: box_result,
                                                      "confidence": confidences[i]}
                        # if need to check the bound boxes's output, draw the bounding boxes on the output image
                        if self.hyperparams['output_to_tmp_dir']:
                            each_image = self._draw_bounding_box(each_image, round(x), round(y), round(x + w), round(y + h))
            else:
                # if nothing detected, still need to output something
                self.logger.warning("Nothing detected on produce image {}".format(each_image_name))
                bbox_count += 1
                output_df_dict[bbox_count] = {"d3mIndex": each_row["d3mIndex"], self._target_column_name: "", "confidence": 0}

            if self.hyperparams['output_to_tmp_dir']:
                self._output_image(each_image, each_image_name)

        output_df = pd.DataFrame.from_dict(output_df_dict, orient='index')
        return output_df

    def _produce_for_retrain_weights(self, input_df):
        """
            produce function for retraining the model weights
        """
        bbox_count = 0
        memo = set()
        output_df_dict = {}
        for i, each_row in input_df.iterrows():
            each_image_name = each_row[self._input_image_column_name]
            if each_image_name in memo:
                continue
            else:
                memo.add(each_image_name)
            self.logger.debug("Predicting on {}".format(each_image_name))
            image_path = os.path.join(self._location_base_uris, each_image_name)
            each_image = cv2.imread(image_path)
            each_image = cv2.cvtColor(each_image, cv2.COLOR_BGR2RGB)
            # Predict Process
            image_size = each_image.shape[:2]
            image_data = self._yolo_utils.image_preporcess(np.copy(each_image), [self.hyperparams["blob_output_shape_x"],
                                                                                 self.hyperparams["blob_output_shape_y"]])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            pred_bbox = self._model.predict(image_data)
            if len(pred_bbox) == 6:
                pred_bbox = pred_bbox[1::2]
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = self._yolo_utils.postprocess_boxes(pred_bbox, image_size, self.hyperparams["blob_output_shape_x"],
                                                        self.hyperparams["confidences_threshold"])
            bboxes = self._yolo_utils.nms(bboxes, self.hyperparams["nms_threshold"], method='nms')

            if len(bboxes) > 0:
                for bbox in bboxes:
                    bbox_count += 1
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    score = '%.4f' % score
                    class_ind = int(bbox[5])
                    xmin, ymin, xmax, ymax = list(coor)
                    box_result = [str(xmin), str(ymin), str(xmin), str(ymax), str(xmax), str(ymax), str(xmax), str(ymin)]
                    box_result = ",".join(box_result)  # remove "[" and "]"
                    # box_result = str(x)+","+str(y) + "," +str(x+w)+ ","+str(y+h)
                    output_df_dict[bbox_count] = {"d3mIndex": each_row["d3mIndex"], self._target_column_name: box_result,
                                                  "confidence": score}
                    # if need to check the bound boxes's output, draw the bounding boxes on the output image
                    if self.hyperparams['output_to_tmp_dir']:
                        each_image = self._draw_bounding_box(each_image, xmin, ymin, xmax, ymax)
            else:
                # if nothing detected, still need to output something
                bbox_count += 1
                self.logger.warning("Nothing detected on produce image {}".format(each_image_name))
                output_df_dict[bbox_count] = {"d3mIndex": each_row["d3mIndex"], self._target_column_name: "0,0,0,0,0,0,0,0",
                                              "confidence": 0}
            if self.hyperparams['output_to_tmp_dir']:
                self._output_image(each_image, each_image_name)

        output_df = pd.DataFrame.from_dict(output_df_dict, orient='index')
        return output_df

    def _load_object_names(self) -> None:
        """
            Inner function used to load default detected object class names with default ight model
        """
        from .yolov3_default_classes import detected_object_names
        self._object_names = detected_object_names.name_list

    def _lazy_init(self, phase="train") -> None:
        """
            a lazy init function which initialize the model only when the primitive's fit/ produce method is called
        """
        if self._inited and self._current_phase == phase:
            return

        if self.hyperparams['use_fitted_weight']:
            self.logger.info("Getting weights file and config file from static volumes ...")
            if "yolov3.weights" not in self.volumes:
                raise ValueError("Can't get weights file!")

            # self._model = self._create_model("test")
            yolov3_cfg_file = os.path.join(os.path.dirname(__file__), 'yolov3.cfg')
            self._model = cv2.dnn.readNet(self.volumes['yolov3.weights'], yolov3_cfg_file)
            self._load_object_names()
            self._output_layer = self._get_output_layers(self._model)
            self.logger.info("Model initialize finished.")

        else:
            self._yolo_dataset_model = importlib.import_module('dsbox.datapreprocessing.featurizer.image.yolo_utils.core.dataset')
            self._yolov3_model = importlib.import_module('dsbox.datapreprocessing.featurizer.image.yolo_utils.core.yolov3')
            self._yolo_utils = importlib.import_module('dsbox.datapreprocessing.featurizer.image.yolo_utils.core.utils')
            self.logger.info("Using customized model...")
            self._model = self._create_model(phase)

        self._current_phase = phase
        self._inited = True

    def _get_output_layers(self, net) -> typing.List[str]:
        """
            function to get the output layer names in the architecture
        """
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def _draw_bounding_box(self, img, x, y, x_plus_w, y_plus_h) -> np.ndarray:
        """
            function to draw the bounding box on given image
        """
        color = [0, 0, 255]
        label = self._object_names[self._target_class_id[0]] if self.hyperparams["use_fitted_weight"] else "target"
        # draw rectangle and put text
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def _output_image(self, image, image_name) -> None:
        """
            function to output the image on D3MLOCALDIR from environment
        """
        output_loc = os.path.join(os.environ.get('D3MLOCALDIR', "/tmp"), image_name)
        cv2.imwrite(output_loc, image)

    def _create_model(self, phase: str = "train"):
        """
            function to create the DNN model for training
        """
        if phase == "train":
            self.optimizer = tf.keras.optimizers.Adam()

        input_layer = tf.keras.layers.Input([self.hyperparams['blob_output_shape_x'], self.hyperparams['blob_output_shape_y'], 3])
        feature_maps = self._yolov3_model.YOLOv3(input_layer)
        output_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = self._yolov3_model.decode(fm, i)
            if phase == "train":
                output_tensors.append(fm)
            output_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, output_tensors)
        if phase == "test":
            if self._dump_model_path is None:
                self._dump_model_path = os.path.join(os.environ.get("D3MLOCALDIR", "/tmp"), "yolov3")
            if not os.path.exists(self._dump_model_path + ".index"):
                raise ValueError("Yolo trained weight file not exist at {}".format(str(self._dump_model_path)))
            self.logger.info("Loading weights from {}".format(self._dump_model_path))
            model.load_weights(self._dump_model_path)
        return model

    def retrain(self) -> None:
        """
            main function that aimed to run fit on pretrained models
        """
        if self._loaded_dataset is None:
            self._load_dataset("train", self._training_inputs, self._training_outputs)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # create check point manager
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self._model)
        ckpt_store_loc = os.path.join(os.environ.get("D3MLOCALDIR", "/tmp"), "tf_checkpoints" + now)
        manager = tf.train.CheckpointManager(ckpt, ckpt_store_loc, max_to_keep=3)

        for i in range(self.hyperparams["epochs"]):
            self.logger.info("Running on No.{} epoch.".format(str(i)))
            for image_data, target in self._loaded_dataset:
                loss = self._train_step(image_data, target)
                if loss.numpy() is np.nan:
                    self.logger.warning("NaN value detected on loss! Roll back to last saved model weights and continue")
                    ckpt.restore(manager.latest_checkpoint)
                else:
                    ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    self.logger.debug("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        # save the retrained model weights after fit
        save_path = os.path.join(os.environ.get("D3MLOCALDIR", "/tmp"), "yolov3_" + now)
        self._dump_model_path = save_path
        self._model.save_weights(save_path)

    def _train_step(self, image_data, target) -> float:
        """
            each train epoch
        """
        with tf.GradientTape() as tape:
            pred_result = self._model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self._yolov3_model.compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            gradients = tape.gradient(total_loss, self._model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
            # set to warning for debug purpose
            self.logger.warning("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                                "prob_loss: %4.2f   total_loss: %4.2f" % (self.global_steps, self.optimizer.lr.numpy(),
                                                                          giou_loss, conf_loss,
                                                                          prob_loss, total_loss))
            # update learning rate
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * self.hyperparams["lr_init"]
            else:
                lr = self.hyperparams["lr_end"] + 0.5 * (self.hyperparams["lr_init"] - self.hyperparams["lr_end"]) * (
                    (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi))
                )
            self.optimizer.lr.assign(lr.numpy())
        return total_loss

    def _load_dataset(self, phase="train", input_df: container.DataFrame = None, output_df: container.DataFrame = None):
        """
            function used to load the input dataset to a proper format for running in yolo
        """
        annotation_list = self._input_to_annotation_list(input_df, output_df, phase=phase)
        dataset_config = {
            "annot_list": annotation_list,
            "input_size": [self.hyperparams['blob_output_shape_x']],
            "batch_size": self.hyperparams['train_batch_size'] if phase == "train" else self.hyperparams['test_batch_size'],
            "data_aug": self.hyperparams['data_augmentation'] if phase == "train" else False,
        }
        self._loaded_dataset = self._yolo_dataset_model.Dataset(dataset_config)
        steps_per_epoch = len(self._loaded_dataset)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = self.hyperparams["warmup_epochs"] * steps_per_epoch
        self.total_steps = self.hyperparams["epochs"] * steps_per_epoch

    def _input_to_annotation_list(self, input_df, output_df, phase="train"):
        """
            function used to transfer the input dataframe to one line style like:
            image_name.jpg bbox1_x_min,bbox1_y_min,bbox1_x_max,bbox1_y_max,object1_id bbox2_x_min,bbox2_y_min,bbox2_x_max,bbox2_y_max,object2_id
            if in test, no bbox will given

        """
        res = []
        # for train, append as described
        if phase == "train":
            merged_df = pd.concat([input_df, output_df], axis=1)
            # drop dulicate columns
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            memo = collections.defaultdict(list)
            for i, each_row in merged_df.iterrows():
                each_line = ""
                each_img = os.path.join(self._location_base_uris, each_row[self._input_image_column_name])
                bbox = each_row[self._target_column_name]
                if isinstance(bbox, str):
                    bbox = bbox.split(",")
                bbox_x = [int(val) for val in bbox[::2]]
                bbox_y = [int(val) for val in bbox[1::2]]
                memo[each_img].append(",".join([str(min(bbox_x)), str(min(bbox_y)), str(max(bbox_x)), str(max(bbox_y)), "0"]))
            for k, v in memo.items():
                res.append(" ".join([k] + v))

        # for test, only append each image's path
        elif phase == "test":
            for i, each_row in input_df.iterrows():
                each_img = os.path.join(self._location_base_uris, each_row[self._input_image_column_name])
                res.append(each_img)

        # random shuffle the results for better diversity
        np.random.shuffle(res)
        return res

    def _cut_image(self, image: np.ndarray, input_shape: typing.List[typing.Union[int, str]]) -> np.ndarray:
        # example of input_shape:
        # input_shape =  ['160', '182', '160', '431', '302', '431', '302', '182']
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners_np = []
        x_axis = []
        y_axis = []
        for i in range(0, len(input_shape), 2):
            roi_corners_np.append((input_shape[i], input_shape[i + 1]))
            x_axis.append(int(input_shape[i]))
            y_axis.append(int(input_shape[i + 1]))
        roi_corners = np.array([roi_corners_np], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex

        # apply the mask
        masked_image = cv2.bitwise_and(image, mask)
        masked_image = masked_image[min(y_axis):max(y_axis), min(x_axis):max(x_axis), :]
        return masked_image

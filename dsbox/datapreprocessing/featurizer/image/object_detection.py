import logging
import numpy as np
import os
import shutil
import sys
import typing
import cv2
import wget
import copy
import collections
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import container

from . import config

logger = logging.getLogger(__name__)

# Input should from dataframe_to_tensor primitive
Inputs = container.DataFrame
Outputs = container.DataFrame  # results


class YoloHyperparams(hyperparams.Hyperparams):
    use_fitted_weight = hyperparams.UniformBool(
        default=True,
        description="A control parameter to set whether to use the pre-trained model weights or train new model",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )
    output_to_tmp_dir = hyperparams.UniformBool(
        default=False,
        description="whether to output the images with bounding boxes for debugging purpose",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )
    # target_object_names = hyperparams.Set(
    #     elements =
    #     )
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
    # scalar with mean values which are subtracted from channels.
    # Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
    blob_mean_R = hyperparams.Uniform(
        default=0,
        lower=0,
        upper=255,
        description="scalar with mean values which are subtracted from channels - color R",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )
    blob_mean_G = hyperparams.Uniform(
        default=0,
        lower=0,
        upper=255,
        description="scalar with mean values which are subtracted from channels - color G",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )
    blob_mean_B = hyperparams.Uniform(
        default=0,
        lower=0,
        upper=255,
        description="scalar with mean values which are subtracted from channels - color B",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"]
        )
    blob_crop = hyperparams.UniformBool(
        default=False,
        description="flag which indicates whether image will be cropped after resize or not",
        semantic_types=["http://schema.org/Boolean", "https://metadata.datadrivendiscovery.org/types/ControlParameter"]
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

    def __init__(self, *, hyperparams: YoloHyperparams, volumes: typing.Union[typing.Dict[str, str], None]=None) -> None:
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

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
            If using the pre-training model, here we will use this model to detect what inside the bounding boxes from
            training dataset. Then, count the number of the objects detected in each box, we will treat only the objects
            amount number larger than the threshold to be the target that we need to detect in the test part.
        """
        if self._fitted:
            return CallResult[None]

        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        # initialization
        self._lazy_init()
        self._output_layer = self._get_output_layers(self._model)
        scale = self.hyperparams['blob_scale_factor']
        blob_x = self.hyperparams['blob_output_shape_x']
        blob_y = self.hyperparams['blob_output_shape_y']
        mean_R = self.hyperparams['blob_mean_R']
        mean_G = self.hyperparams['blob_mean_G']
        mean_B = self.hyperparams['blob_mean_B']
        crop = self.hyperparams['blob_crop']
        conf_threshold = self.hyperparams['confidences_threshold']
        nms_threshold = self.hyperparams['nms_threshold']
        class_count = [0 for i in range(len(self._object_names))]

        # load images from input dataframe
        image_only = self._training_inputs.drop(columns=['d3mIndex'])
        if len(image_only.columns) > 1:
            logger.warning("Detect multiple file columns inputs! Will only use the first columns as the input image column")
        image_names_list = image_only[image_only.columns[0]].tolist()

        for i, each_image_name in enumerate(image_names_list):

            each_image = cv2.imread(os.path.join(self._location_base_uris, each_image_name))
            ground_truth_box = self._training_outputs.iloc[i,0].split(",")
            logger.debug("processing", each_image_name, "on", ground_truth_box)
            # update 2019.5.9: cut the image into the bounding box area only
            each_image = self._cut_image(each_image, ground_truth_box)
            # Creates 4-dimensional blob from image.
            # swapRB has to be True, otherwise the channel is not R,G,B style
            blob = cv2.dnn.blobFromImage(each_image, scale, (blob_x, blob_y), (mean_R, mean_G, mean_B), True, crop=crop)
            # set input blob for the network
            self._model.setInput(blob)
            outs = self._model.forward(self._output_layer)

            # initialization for each image
            class_ids = []
            confidences = []
            boxes = []
            width = each_image.shape[1]
            height = each_image.shape[0]
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
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            if len(indices) > 0:
                for each in indices:
                    i = each[0]
                    box = boxes[i]
                    class_count[class_ids[i]] += 1
            else:
                logger.warning("No object detected on " + each_image_name + " on [" + self._training_outputs.iloc[i, 0] + "]")

        # find the real target that we need to detect
        for i, each in enumerate(class_count):
            if each >= conf_threshold * len(image_names_list):
                self._target_class_id.append(i)
        if len(self._target_class_id) < 1:
            logger.error("No corresponding target object detected in training set with pre-trained model")
        elif len(self._target_class_id) > 1:
            logger.warning("More than 1 target object detected in the training set with pre-trained model")

        target_id_str = ",".join(self._object_names[x] for x in self._target_class_id)
        logger.info("The target class id is: [" + target_id_str + "]")

        self._fitted = True
        return CallResult(None, has_finished=True, iterations_done=1)

    def get_params(self) -> Params:
        param = Params(
                       target_class_id = self._target_class_id,
                       output_layer = self._output_layer,
                       target_column_name = self._target_column_name,
                      )
        return param

    def set_params(self, *, params: Params) -> None:
        self._target_class_id = params["target_class_id"]
        self._output_layer = params["output_layer"]
        self._target_column_name = params["target_column_name"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        target_column_names = outputs.columns.tolist()
        if len(target_column_names) > 1:
            logger.warn("Multiple target detected!")
        else:
            self._target_column_name = target_column_names[0]

        self._location_base_uris = self._get_image_path(inputs)
        self._fitted = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Use YOLO to detect the objects in the input dataframe
            The function will read the images if the input is a dataframe with image names
            The detected output depends on the input training dataset's ojbects
        """
        results = copy.copy(inputs)
        scale = self.hyperparams['blob_scale_factor']
        blob_x = self.hyperparams['blob_output_shape_x']
        blob_y = self.hyperparams['blob_output_shape_y']
        mean_R = self.hyperparams['blob_mean_R']
        mean_G = self.hyperparams['blob_mean_G']
        mean_B = self.hyperparams['blob_mean_B']
        crop = self.hyperparams['blob_crop']
        conf_threshold = self.hyperparams['confidences_threshold']
        nms_threshold = self.hyperparams['nms_threshold']
        self._lazy_init()
        self._location_base_uris = self._get_image_path(inputs)
        image_d3mIndex = results['d3mIndex'].astype(int).tolist()
        image_only = results.drop(columns=['d3mIndex'])
        file_name_column = image_only.columns.tolist()[0]
        if len(image_only.columns) > 1:
            logger.warning("Detect multiple file columns inputs! Will try to use the first columns as the input image column")
        image_names_list = image_only[image_only.columns[0]].tolist()
        dict_image_name_to_d3mIndex = {}
        for i, v in results.iterrows():
            dict_image_name_to_d3mIndex[v[file_name_column]] = v["d3mIndex"]

        object_count_in_each_image = collections.defaultdict(int)
        for each in image_names_list:
            object_count_in_each_image[each] += 1

        output_dataFrame = container.DataFrame(columns = ["d3mIndex", self._target_column_name, "confidence"])

        for i, each_image_name in enumerate(object_count_in_each_image.keys()):
            image_path = os.path.join(self._location_base_uris, each_image_name)
            each_image = cv2.imread(image_path)
            if each_image is None:
                logger.error("loading image from " + image_path + " failed!")
                continue
            logger.debug("Now detecting objects in", each_image_name)
            # Creates 4-dimensional blob from image.
            # swapRB has to be True, otherwise the channel is not R,G,B style
            blob = cv2.dnn.blobFromImage(each_image, scale, (blob_x, blob_y), (mean_R, mean_G, mean_B), True, crop=crop)
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
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            found_object_amount = 0
            # if len(indices) > object_count_in_each_image[each_image_name]:
                # logger.warn(str(len(indices)) + " objects detected "+ each_image_name + " while only "+str(object_count_in_each_image[each_image_name])+ " required.")

            for each in indices:
                i = each[0]
                if class_ids[i] in self._target_class_id:# and found_object_amount < object_count_in_each_image[each_image_name]:
                    box = boxes[i]
                    found_object_amount += 1
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    box_result = [str(x), str(y), str(x), str(y+h), str(x+w), str(y+h), str(x+w), str(y)]
                    box_result = ",".join(box_result) # remove "[" and "]"
                    # box_result = str(x)+","+str(y) + "," +str(x+w)+ ","+str(y+h)
                    output_dataFrame = output_dataFrame.append({"d3mIndex":dict_image_name_to_d3mIndex[each_image_name], self._target_column_name: box_result, "confidence":confidences[i]}, ignore_index=True)
                    # if need to check the bound boxes's output, draw the bounding boxes on the output image
                    if self.hyperparams['output_to_tmp_dir']:
                        each_image = self._draw_bounding_box(each_image, round(x), round(y), round(x+w), round(y+h))

            if self.hyperparams['output_to_tmp_dir']:
                self._output_image(each_image, each_image_name)

            # add empty columns if found less amount of objects than except to ensure length of indexes are same
            # while found_object_amount < object_count_in_each_image[each_image_name]:
                # output_dataFrame = output_dataFrame.append({"d3mIndex":dict_image_name_to_d3mIndex[each_image_name], self._target_column_name:"", confidence:"0"}, ignore_index=True)
                # found_object_amount += 1

        metadata_selector = (metadata_base.ALL_ELEMENTS, 0)
        output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=results.metadata.query(metadata_selector), selector=metadata_selector)
        # add prediction column's metadata
        for each_column in range(1, output_dataFrame.shape[1]):
            metadata_selector = (metadata_base.ALL_ELEMENTS, each_column)
            metadata_each_column = {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PredictedTarget',)}
            output_dataFrame.metadata = output_dataFrame.metadata.update(metadata=metadata_each_column, selector=metadata_selector)

        self._has_finished = True
        self._iterations_done = True
        return CallResult(output_dataFrame, self._has_finished, self._iterations_done)

    def _load_object_names(self) -> None:
        """
            Inner function used to load default detected object class names with default ight model
        """
        from .yolov3_default_classes import detected_object_names
        self._object_names = detected_object_names.name_list

    def _lazy_init(self) -> None:
        """
            a lazy init function which initialize the model only when the primitive's fit/ produce method is called
        """
        if self._inited:
            return
        if self.hyperparams['use_fitted_weight']:
            logger.info("Getting weights file and config file from static volumes ...")

            if "yolov3.weights" not in self.volumes:
                raise ValueError("Can't get weights file!")

            yolov3_cfg_file = os.path.join(os.path.dirname(__file__), 'yolov3.cfg')
            self._model = cv2.dnn.readNet(self.volumes['yolov3.weights'], yolov3_cfg_file)
            self._load_object_names()
            logger.info("Model initialize finished.")

        else:
            logger.info("Using customized model...")
            logger.info("This function not finished yet.")
            # self._model =

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
        color = [0,0,255]
        label = self._object_names[self._target_class_id[0]]
        # draw rectangle and put text
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def _output_image(self, image, image_name) -> None:
        """
            function to output the image on D3MLOCALDIR from environment
        """
        output_loc = os.path.join(os.environ['D3MLOCALDIR'], image_name)
        cv2.imwrite(output_loc, image)

    def _get_image_path(self, dataset_input) -> str:
        """
            function used to get the abs path of input images
        """
        target_index = []
        location_base_uris = []
        elements_amount = dataset_input.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        for selector_index in range(elements_amount):
            each_selector = dataset_input.metadata.query((metadata_base.ALL_ELEMENTS, selector_index))
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
            logger.error("[ERROR] No image related column found!")
        elif len(target_index) > 1:
            logger.info("[INFO] Multiple image columns found in the input, this primitive can only handle one column.")
        location_base_uris = location_base_uris[0]
        if location_base_uris[0:7] == 'file://':
            location_base_uris = location_base_uris[7:]
        return location_base_uris

    def _cut_image(self, image:np.ndarray, input_shape:typing.List[typing.Union[int, str]]) -> np.ndarray:
        # example of input_shape:
        # input_shape =  ['160', '182', '160', '431', '302', '431', '302', '182']
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners_np = []
        x_axis = []
        y_axis = []
        for i in range(0,len(input_shape),2):
            roi_corners_np.append((input_shape[i], input_shape[i+1]))
            x_axis.append(int(input_shape[i]))
            y_axis.append(int(input_shape[i+1]))
        roi_corners = np.array([roi_corners_np], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # from Masterfool: use cv2.fillConvexPoly if you know it's convex

        # apply the mask
        masked_image = cv2.bitwise_and(image, mask)
        masked_image = masked_image[min(y_axis):max(y_axis),min(x_axis):max(x_axis),:]
        return masked_image

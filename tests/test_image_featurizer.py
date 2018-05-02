import os
import numpy as np
import unittest

from keras.preprocessing import image as keras_image

from dsbox.datapreprocessing.featurizer.image import ResNet50ImageFeature, Vgg16ImageFeature, Vgg16Hyperparams, ResNet50Hyperparams

class TestImageFeaturizer(unittest.TestCase):
    def setUp(self):
        image_list = []
        image_dir = os.path.join(os.path.dirname(__file__), 'images')
        for filename in os.listdir(image_dir):
            filepath = os.path.join(image_dir, filename)
            image_list.append(keras_image.load_img(filepath, target_size=(224, 224)))

        # Convert to tensor
        shape = (len(image_list), ) + keras_image.img_to_array(image_list[0]).shape
        self.image_tensor = np.empty(shape)
        for i in range(len(image_list)):
            self.image_tensor[i] = keras_image.img_to_array(image_list[i])

    def test_vgg(self):
        uniform_int = Vgg16Hyperparams.configuration['layer_index']
        num_images = self.image_tensor.shape[0]
        layer_size = [25088, 100352, 200704, 401408]
        for index in [uniform_int.lower, (uniform_int.lower + uniform_int.upper) // 2, uniform_int.upper-1]:
            vgg = Vgg16ImageFeature(hyperparams={'layer_index': index})
            result = vgg.produce(inputs=self.image_tensor)
            self.assertEqual(result.value.shape, (num_images, layer_size[index]), 'layer_index={}'.format(index))

    def test_res(self):
        uniform_int = ResNet50Hyperparams.configuration['layer_index']
        num_images = self.image_tensor.shape[0]
        layer_size = [2048, 100352, 25088, 25088, 100352, 25088, 25088, 100352, 25088, 25088, 200704]
        for index in [uniform_int.lower, (uniform_int.lower + uniform_int.upper) // 2, uniform_int.upper-1]:
            res = ResNet50ImageFeature(hyperparams={'layer_index': index})
            result = res.produce(inputs=self.image_tensor)
            self.assertEqual(result.value.shape, (num_images, layer_size[index]), 'layer_index={}'.format(index))

if __name__ == '__main__':
    unittest.main()

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        import keras.applications.resnet50 as resnet50
        import keras.applications.vgg16 as vgg16
        resnet50.ResNet50(weights='imagenet')
        vgg16.VGG16(weights='imagenet', include_top=False)
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        import keras.applications.resnet50 as resnet50
        import keras.applications.vgg16 as vgg16
        resnet50.ResNet50(weights='imagenet')
        vgg16.VGG16(weights='imagenet', include_top=False)
        install.run(self)

setup(name='dsbox-featurizer',
      version='0.1.4',
      url='https://github.com/usc-isi-i2/dsbox-featurizer',
      maintainer_email='fanghaol@usc.edu',
      maintainer='Fanghao Luo',
      description='DSBox Featurization primitives',
      license='MIT',
      packages=['dsbox', 'dsbox.datapreprocessing', 'dsbox.datapreprocessing.featurizer'
      , 'dsbox.datapreprocessing.featurizer.multiTable'
      , 'dsbox.datapreprocessing.featurizer.image'
      , 'dsbox.datapreprocessing.featurizer.pass'
      , 'dsbox.datapreprocessing.featurizer.timeseries'],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'scipy>=0.19.0,<1.2', 'numpy>=1.11.1', 'pandas>=0.20.1',
          'python-dateutil>=2.5.2', 'six>=1.10.0', 'stopit==1.1.2',
          'scikit-learn>=0.18.0',
          'Keras<=2.1.6,>=2.0.2', 'Pillow==5.1.0', 'tensorflow==1.8.0', 'h5py<=2.7.1',
          'dsbox-datacleaning==1.3.0'
      ],
      dependency_links=[
        'git+https://github.com/usc-isi-i2/dsbox-cleaning@c1c284123ba12a07b5f24a3e9f809e75d1b8be42#egg=dsbox-datacleaning-1.3.0'
      ],
      keywords='d3m_primitive',
      entry_points = {
          'd3m.primitives': [
              'dsbox.DoNothing = dsbox.datapreprocessing.featurizer.pass:DoNothing',
              'dsbox.MultiTableFeaturization = dsbox.datapreprocessing.featurizer.multiTable:MultiTableFeaturization',
              'dsbox.DataFrameToTensor = dsbox.datapreprocessing.featurizer.image:DataFrameToTensor',
              'dsbox.Vgg16ImageFeature = dsbox.datapreprocessing.featurizer.image:Vgg16ImageFeature',
              'dsbox.ResNet50ImageFeature = dsbox.datapreprocessing.featurizer.image:ResNet50ImageFeature',
              'dsbox.TimeseriesToList = dsbox.datapreprocessing.featurizer.timeseries:TimeseriesToList',
              'dsbox.RandomProjectionTimeSeriesFeaturization = dsbox.datapreprocessing.featurizer.timeseries:RandomProjectionTimeSeriesFeaturization'
          ],
      },
      cmdclass={
          'develop': PostDevelopCommand,
          'install': PostInstallCommand,
      }
)

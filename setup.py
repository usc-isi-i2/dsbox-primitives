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
      version='1.0.2',
      url='https://github.com/usc-isi-i2/dsbox-featurizer',
      maintainer_email='Ke-Thia Yao',
      maintainer='kyao@isi.edu',
      description='DSBox Featurization primitives',
      license='MIT',
      packages=[
          'dsbox',
          'dsbox.datapreprocessing',
          'dsbox.datapreprocessing.featurizer',
          'dsbox.datapreprocessing.featurizer.multiTable',
          'dsbox.datapreprocessing.featurizer.image',
          'dsbox.datapreprocessing.featurizer.pass',
          'dsbox.datapreprocessing.featurizer.timeseries'],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'scipy>=0.19.0,<1.2', 'numpy>=1.11.1', 'pandas>=0.20.1',
          'python-dateutil>=2.5.2', 'six>=1.10.0', 'stopit==1.1.2',
          'scikit-learn>=0.18.0','wget',
          'Keras==2.2.4', 'Pillow==5.1.0', 'tensorflow==1.12', 'h5py<=2.7.1', "pyramid-arima==0.8.1"
      ],
      # dependency_links=[
      #   'git+https://github.com/usc-isi-i2/dsbox-cleaning.git@7106cf48d56eaa45460792edf7416dfd57548370#egg=dsbox-datacleaning-1.3.1'
      # ],
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'data_preprocessing.do_nothing.DSBOX = dsbox.datapreprocessing.featurizer.pass:DoNothing',
              'data_preprocessing.do_nothing_for_dataset.DSBOX = dsbox.datapreprocessing.featurizer.pass:DoNothingForDataset',
              'feature_extraction.multitable_featurization.DSBOX = dsbox.datapreprocessing.featurizer.multiTable:MultiTableFeaturization',
              'data_preprocessing.dataframe_to_tensor.DSBOX = dsbox.datapreprocessing.featurizer.image:DataFrameToTensor',
              'feature_extraction.yolo.DSBOX = dsbox.datapreprocessing.featurizer.image:Yolo',
              'feature_extraction.vgg16_image_feature.DSBOX = dsbox.datapreprocessing.featurizer.image:Vgg16ImageFeature',
              'feature_extraction.resnet50_image_feature.DSBOX = dsbox.datapreprocessing.featurizer.image:ResNet50ImageFeature',
              'data_preprocessing.time_series_to_list.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:TimeseriesToList',
              'feature_extraction.random_projection_timeseries_featurization.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:RandomProjectionTimeSeriesFeaturization',
              'data_transformation.group_up_by_timeseries.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:GroupUpByTimeSeries',
              'time_series_forecasting.arima.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:AutoArima',
              'time_series_forecasting.rnn_time_series.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:RNNTimeSeries',

          ],
      },
      cmdclass={
          'develop': PostDevelopCommand,
          'install': PostInstallCommand,
      }
)

from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_requires = list()
    dependency_links = list()
    for line in f:
        re = line.strip()
        if re:
            install_requires.append(re)

setup(name='dsbox-primitives',
      version='1.5.3',
      description='DSBox data processing primitives for both cleaning and featurizer',
      author='USC ISI',
      url='https://github.com/usc-isi-i2/dsbox-primitives.git',
      maintainer_email='kyao@isi.edu',
      maintainer='Ke-Thia Yao',
      license='MIT',
      packages=[
                'dsbox',
                'dsbox.datapreprocessing',
                'dsbox.datapreprocessing.cleaner',
                'dsbox.datapostprocessing',
                'dsbox.datapreprocessing.featurizer',
                'dsbox.datapreprocessing.featurizer.multiTable',
                'dsbox.datapreprocessing.featurizer.image',
                'dsbox.datapreprocessing.featurizer.pass',
                'dsbox.datapreprocessing.featurizer.timeseries'
               ],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'classification.ensemble_voting.DSBOX = dsbox.datapostprocessing:EnsembleVoting',
              'classification.lstm.DSBOX = dsbox.datapreprocessing.featurizer.image:LSTM',
              'data_cleaning.cleaning_featurizer.DSBOX = dsbox.datapreprocessing.cleaner:CleaningFeaturizer',
              # 'data_cleaning.column_fold.DSBOX = dsbox.datapreprocessing.cleaner:FoldColumns',
              'data_cleaning.label_encoder.DSBOX = dsbox.datapreprocessing.cleaner:Labler',
              'data_preprocessing.dataframe_to_tensor.DSBOX = dsbox.datapreprocessing.featurizer.image:DataFrameToTensor',
              'data_preprocessing.do_nothing.DSBOX = dsbox.datapreprocessing.featurizer.pass:DoNothing',
              'data_preprocessing.do_nothing_for_dataset.DSBOX = dsbox.datapreprocessing.featurizer.pass:DoNothingForDataset',
              'data_preprocessing.encoder.DSBOX = dsbox.datapreprocessing.cleaner:Encoder',
              'data_preprocessing.greedy_imputation.DSBOX = dsbox.datapreprocessing.cleaner:GreedyImputation',
              'data_preprocessing.horizontal_concat.DSBOX = dsbox.datapostprocessing:HorizontalConcat',
              'data_preprocessing.iterative_regression_imputation.DSBOX = dsbox.datapreprocessing.cleaner:IterativeRegressionImputation',
              'data_preprocessing.mean_imputation.DSBOX = dsbox.datapreprocessing.cleaner:MeanImputation',
              'data_preprocessing.splitter.DSBOX = dsbox.datapreprocessing.cleaner:Splitter',
              'data_preprocessing.time_series_to_list.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:TimeseriesToList',
              'data_preprocessing.unary_encoder.DSBOX = dsbox.datapreprocessing.cleaner:UnaryEncoder',
              # 'data_preprocessing.unfold.DSBOX = dsbox.datapostprocessing:Unfold',
              'data_preprocessing.vertical_concatenate.DSBOX = dsbox.datapostprocessing:VerticalConcat',
              'data_transformation.to_numeric.DSBOX = dsbox.datapreprocessing.cleaner:ToNumeric',
              'feature_extraction.inceptionV3_image_feature.DSBOX = dsbox.datapreprocessing.featurizer.image:InceptionV3ImageFeature',
              'feature_extraction.multitable_featurization.DSBOX = dsbox.datapreprocessing.featurizer.multiTable:MultiTableFeaturization',
              'feature_extraction.random_projection_timeseries_featurization.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:RandomProjectionTimeSeriesFeaturization',
              'feature_extraction.resnet50_image_feature.DSBOX = dsbox.datapreprocessing.featurizer.image:ResNet50ImageFeature',
              'feature_extraction.vgg16_image_feature.DSBOX = dsbox.datapreprocessing.featurizer.image:Vgg16ImageFeature',
              'feature_extraction.yolo.DSBOX = dsbox.datapreprocessing.featurizer.image:Yolo',
              'normalization.iqr_scaler.DSBOX = dsbox.datapreprocessing.cleaner:IQRScaler',
              'schema_discovery.profiler.DSBOX = dsbox.datapreprocessing.cleaner:Profiler',
              'time_series_forecasting.arima.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:AutoArima',
              # 'time_series_forecasting.rnn_time_series.DSBOX = dsbox.datapreprocessing.featurizer.timeseries:RNNTimeSeries',
          ],
      })

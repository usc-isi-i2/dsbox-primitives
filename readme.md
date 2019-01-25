# ISI DSBox Featurization Primitives

The git repository for DSBox primitives related to featurization is located [here](https://github.com/usc-isi-i2/dsbox-featurizer). The git repository containing DSBox cleaning related primitives is [here](https://github.com/usc-isi-i2/dsbox-cleaning).

## Image Featurization Primitives

### d3m.primitives.dsbox.ResNet50ImageFeature

Generate features using pre-trained ResNet50 deep neural network. Use hyperparameter `layer_index` to select the network layer to use for featurization.

### d3m.primitives.dsbox.Vgg16ImageFeature

Generate features using pre-trained VGG16 deep neural network. Use hyperparameter `layer_index` to select the network layer to use for featurization.

### d3m.primitives.dsbox.DataFrameToTensor

Reads in image files and generates a tensor that suitable as input to `d3m.primitives.dsbox.ResNet50ImageFeature` and `d3m.primitives.dsbox.Vgg16ImageFeature`.

## Timeseries Featuration Primitives

### d3m.primitives.dsbox.RNNTimeSeries

Performs forecasting of one timeseries using recursive neural network.

### d3m.primitives.dsbox.AutoArima

Performs forecasting of one timeseries using AutoArima.

### d3m.primitives.dsbox.GroupUpByTimeSeries

Performs forecasting of one timeseries using Group Up.

### d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization

Generate features of multiple timeseries by random projecting the timeseries matrix into lower dimendions.

### d3m.primitives.dsbox.TimeseriesToList

Reads in timeseries csv files and generate output List that is suitable as input to `d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization`.

## Multi-table Join Primitive

### d3m.primitives.dsbox.MultiTableFeaturization

Automatically detect foriegn key relationships among multiple tables, and join the tables into one table using aggregation.

## Miscellaneous

### d3m.primitives.dsbox.DoNothing

This an identity function primitive that returns the input dataframe as output. This useful for bypassing a step in a pipeline without having to modify the pipeline structure.

![travis ci](https://travis-ci.org/usc-isi-i2/dsbox-cleaning.svg?branch=devel)
# ISI Primitives

# ISI DSBox Cleaning Primitives

The git repository containing DSBox cleaning related primitives is [here](https://github.com/usc-isi-i2/dsbox-cleaning). The git repository for DSBox primitives related to featurization is located [here](https://github.com/usc-isi-i2/dsbox-featurizer).

## Data cleaning primitives

### d3m.primitives.dsbox.CleaningFeaturizer

This is a multi-purpose cleaning featurizer primitive. This primitive requires metadata annotations from ISI's profiling primitive, see `d3m.primitives.dsbox.Profiler` below. The cleaning featurization operations supported include:

* Split date column into multiple columns, e.g. year, month, date, day
* Split US phone number into multiple columns.
* Split column with consistent alpha-numeric value patterns, e.g. '2days' into multiple columns.
* Split column with consistent puntucation value patterns, e.g. 'NY_US' into multiple columns.

### d3m.primitives.dsbox.FoldColumns

Fold multiple columns into one column based on common column name prefix. For example, fold columns with names 'month-jan', 'month-feb', 'month-mar' and so on, into one column named 'month'.

## Encoding primitives

### d3m.primitives.dsbox.Encoder

Performs one-hot encoding for categorical attributes. This encoder can handle missing values, and it allows user to specify the upper limit of columns to generate per cagtegorical attribute, `n_limit`.

### d3m.primitives.dsbox.UnaryEncoder

Performs unary encoding, which useful for ordinal data.

## Imputation primitives

### d3m.primitives.dsbox.MeanImputation

Performs mean missing value imputation for numerical columns, and mode imputation for categorical columns.

### d3m.primitives.dsbox.GreedyImputation

Performs missing value imputation by greedy search over simple imputation methods, i.e. mean, min, max, and zero.

### d3m.primitives.dsbox.IterativeRegressionImputation

Performs missing value imputation by regression, then improve the imputation by iterating over columns with missing values.

## Profiling Primitive

### d3m.primitives.dsbox.Profiler

This primitive generates metadata by examining the given data. The types of metadata include:

* Column contains values tokenizable as an American phone number
* Column contains values tokenizable by puntucation
* Column contains values tokenizable into numerical tokens and alpha tokens
* Column value tokenization features (most common tokens, number of distinct tokens, ratio of distinct tokens, and so on)
* Column value features (most common values, number of distinct values, ration of distinct values, and so on)
* Column contains filename-like values
* Column contains missing values (number of missing values, ratio of missing values)
* Number of outlier values
* Correlation between columns (Pearson, Spearman)

## Datamart Primitives

### d3m.primitives.dsbox.QueryDataframe

Queries datamart for available datasets. The JSON query specification is defined [Datamart Query API](https://datadrivendiscovery.org/wiki/display/work/Datamart+Query+API "Datamart Query API "). The primitive returns a list of dataset metadata.

### d3m.primitives.dsbox.Join

Joins two dataframes into one dataframe. The primtive takes two dataframes, left\_dataframe and right\_dataframe, and two lists specifing the join columns, left\_columns and right\_columns.


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


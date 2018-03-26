## install

according to [this issue](https://gitlab.com/datadrivendiscovery/metadata/issues/30)

install it in edit mode:

'''
pip install -e git+https://github.com/usc-isi-i2/dsbox-featurizer.git#egg=dsbox-featurizer
'''


## pipeline

follow: [nist/ta1-pipeline-submission](https://gitlab.datadrivendiscovery.org/nist/ta1-pipeline-submission/tree/master)

## Multiple Table Featurizer

draft version.

tested datasets:


* [Financial](https://relational.fit.cvut.cz/dataset/Financial)
* [Mutagenesis](https://relational.fit.cvut.cz/dataset/Mutagenesis)


## Results
example results, on `uu3_world_development_indicators` dataset; compared with un-featurized data (only use master table)

5-fold cross validation results are listed; using "mean_squared_error" as metric

first version:

```
using ExtraTreeRegressor:
====> for featurized data:
[-12.46043294  -6.95689044  -3.73603532  -3.71090919  -8.01907025]
====> for original data:
[ -93.01861321  -29.19345933 -119.48080097  -53.20908888  -36.69043778]
using LinearRegression:
====> for featurized data:
[-42.19727895 -30.07312712 -22.37585192 -24.23403244 -26.36721586]
====> for original data:
[-133.30071736 -110.3284196   -93.93481927  -98.30471911  -81.73845532]
```


as can be found, using our featurized data (aggregate with other tables) leads to much better result.


on `dev` branch: using multiple aggregation functions to numeric attributes:

```
using ExtraTreeRegressor:    
====> for featurized data:   
[-12.55563377  -7.68476608  -7.47071635  -3.77852268  -8.08901999]                                                    
====> for original data:     
[-45.47443448 -11.21202864 -18.22413694  -9.25713625 -85.75955098]                                                    
using LinearRegression:      
====> for featurized data:   
[-196.39653237 -116.13720674  -86.94102942 -103.79982623 -134.2683943 ]                                               
====> for original data:     
[-133.30071736 -110.3284196   -93.93481927  -98.30471911  -81.73845532] 
```

using more numeric aggregation functions does not lead to better performance  for this dataset.
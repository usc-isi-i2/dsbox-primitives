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

### on `master` branch:

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

#### Analysis
obviously some aggregated information helps, what are they? We could see them using the coefficients of LienarRegression model. Belowing are the attributes that with largest coefficients (absolute value) of our featurized data:

```
[('Country.csv_NationalAccountsReferenceYear_2013/14', 1.7499951007985628), ('Country.csv_Region_North America', 1.8704838587929624), ('Country.csv_IncomeGroup_Upper middle income', 1.9431183647373362), ('Country.csv_VitalRegistrationComplete_Yes', 1.9613746051136511), ('Country.csv_SystemOfTrade_other_', 2.027561008601204), ('Country.csv_PppSurveyYear_2011 (household consumption only).', 2.205419322428428), ('Country.csv_IncomeGroup_High income: OECD', 2.6628945666285846), ('Country.csv_Region_Latin America & Caribbean', 3.9492457890565063), ('Country.csv_NationalAccountsReferenceYear_2013', 4.037421825654523), ('Country.csv_LendingCategory_other_', 5.3837670803742705)]
```

```
[('Country.csv_Region_Sub-Saharan Africa', -5.228651309518281), ('Country.csv_IncomeGroup_Low income', -4.824404961612842), ('Country.csv_VitalRegistrationComplete_other_', -2.5529468758340257), ('Country.csv_LendingCategory_IDA', -2.511902747661453), ('Country.csv_LendingCategory_Blend', -2.3045186056059523), ('Country.csv_NationalAccountsReferenceYear_other_', -2.143242562394621), ('Country.csv_SnaPriceValuation_other_', -1.9487197441571942), ('Country.csv_Region_South Asia', -1.8686526473981844), ('Country.csv_ImfDataDisseminationStandard_other_', -1.819878860294755), ('Country.csv_NationalAccountsReferenceYear_1997', -1.7757587997036153)]
```

as can be seed from the listed attributes above, the result make sense, for example: 'Country.csv_IncomeGroup_Low income' gives negative contribution to the expected life, while 'Country.csv_IncomeGroup_Upper middle income' is in the opposite.


### on `dev` branch: 

using multiple aggregation functions to numeric attributes:

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

using more numeric aggregation functions does not lead to better performance for this dataset.
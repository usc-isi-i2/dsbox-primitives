from dsbox.datapreprocessing.featurizer.multiTable import MultiTableFeaturization
from d3m_metadata.container import DataFrame, List
from d3m_metadata import hyperparams, metadata
from typing import Union

class Hyperparams(hyperparams.Hyperparams):
    pass


from os import listdir
import pandas as pd


# step 1: prepare inputs
data_path = "/Users/luofanghao/work/USC_lab/isi-II/work/DSBox_project/multiple_table/test_data/financial/"
# data_path = "/Users/luofanghao/work/USC_lab/isi-II/work/DSBox_project/multiple_table/test_data/mutagenesis/"
# master_col_name = "molecule.csv_molecule_id"
master_col_name = "loan.csv_account_id" # master table and its primary key
names = listdir(data_path)
tables_names = list(name for name in names) # list of table names


data = List()
names = List()
for x in tables_names:
    data.append(pd.read_csv(data_path + x))
    names.append(x)

print("tables are: {}".format(names))
names.append(master_col_name) # master table_column name

# step 2: use featurizer
featurizer = MultiTableFeaturization(hyperparams=Hyperparams())
result = featurizer.produce(inputs=(data, names))
result.value.to_csv("featureized_example.csv", index=False)

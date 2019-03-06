import pandas as pd
from os import listdir
import time
from d3m import container
import logging
Inputs = container.Dataset
_logger = logging.getLogger(__name__)

def get_relation_matrix(data:Inputs, relations: tuple):
    """
    Parameter: 
    tables_names: list of strings, the other table file names

    Return:
    relation_matrix
    """
    
    start_time = time.time()

    all_tables = data # dict, key: name; value : pandas.DataFrame
    all_tables_col = {} # dict, key: table + "_" + col_name ; value : set of the column
    #counter = 0

    for x in all_tables.keys():
        for col_name in all_tables[x].keys():
            key = x + "_" + col_name
            all_tables_colSet[key] = set(all_tables[x][col_name])

    _logger.info("=====>> data readin finished: {}".format(time.time() - start_time))

    # 1. define the relation_matrix: index and link
    relation_matrix_index = list(all_tables_colSet.keys())  # list of columns name (index)
    # source2index = {}         # dict, key: (table_name, col_name), value: index in matrix
            
    relation_matrix = pd.DataFrame(index=relation_matrix_index, columns=relation_matrix_index)
    # 2. calculate
    # table1 is assumed to be the master table, which will be columns_id in relation_matrix
    for table1 in all_tables:
        for table2 in all_tables:
            start_time = time.time()
            if (table1 == table2): continue
            # compute all column pairs
            for col_name1 in all_tables[table1].keys():
                for col_name2 in all_tables[table2].keys():
                    # col1 = all_tables[table1][col_name1]
                    # col2 = all_tables[table2][col_name2]
                    # i = source2index[(table1, col_name1)]
                    # j = source2index[(table2, col_name2)]
                    i = table1 + "_" + col_name1
                    j = table2 + "_" + col_name2
                    seti = all_tables_colSet[i]
                    setj = all_tables_colSet[j]
                    
                    relation_matrix[i][j] = cal_relation_val_fromset(seti, setj)
            _logger.info("=====>> {} vs {} finished: {}".format(table1, table2, time.time() - start_time))       
    return relation_matrix

def cal_relation_val_fromset(s_i, s_j):
    return len(s_i.intersection(s_j))/float(len(s_i))

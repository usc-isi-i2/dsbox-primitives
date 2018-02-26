import re
import numpy as np

def comp_skewness(all_tables, c1, c2):
    """
    """
    split = re.split(".csv_", c1)
    filename = split[0] + ".csv"
    column = all_tables[filename][split[1]]
    if (not np.issubdtype(column.dtype, np.number)): 
        print ("cannot compute for {}".format(c1))
        return 1 # not numeric value inside
    range1 = column.max() - column.min()
    
    split = re.split(".csv_", c2)
    filename = split[0] + ".csv"
    column = all_tables[filename][split[1]]
    if (not np.issubdtype(column.dtype, np.number)): 
        print ("cannot compute for {}".format(c2))
        return 1 # not numeric value inside
    range2 = column.max() - column.min()
    
    skewness = range1/float(range2)
    if (skewness == 1):
        print ("c1: {} c2: {}".format(c1, c2))
    
    
    return skewness
    

def relationMat2foreignKey(dataset, names, relation_matrix):
    """
    get foreignKey relationship from given relation matrix
    TODO:
        1. optimize time complexity
    """

    all_tables = {} # dict, key: name; value : pandas.DataFrame
    counter = 0
    for x in names:
        all_tables[x] = dataset[counter]
        counter+=1


    index = relation_matrix.keys()
    relation_matrix = relation_matrix.fillna(0)

    result = set()
    print ("found matched col pairs: (foreign key ==> primary key) \n")
    for col_i in index:
        skewness_dict = dict()
        for col_j in index:
            val = relation_matrix[col_i][col_j]
            val_opp = relation_matrix[col_j][col_i]
            if (val == 1):
                # one hard-coded rule appied here: master_col should contains no nan values, no duplicates
                split = re.split(".csv_", col_j)
                filename = split[0] + ".csv"
                data = all_tables[filename]
                if (data[split[1]].duplicated().any() > 0 or data[split[1]].isnull().any()):
                    continue
                # compute skewness of current result
                skewness = comp_skewness(all_tables, col_i, col_j)
                skewness_dict[skewness] = col_j

        if (len(skewness_dict) > 0):
            max_skewness = sorted(skewness_dict.keys())[-1] 
            if (max_skewness < 0.7): continue
            col = skewness_dict[max_skewness]
            result.add((col_i,col))
            print ("==== select {}, from {}, skewness value is {}".format(col, skewness_dict.values(), max_skewness))
            print ("{} ====> {}".format(col_i,col))
    
    return result



# ====================== relation correction function==========================
def relations_correction(relations):
    """
    to correct the obtained relations:
        1. if more than one relation found btw. two tables, only pick one of them
    """
    # using easist way to fix: a set that avoid duplicates
    table_tuple_set = {} # store the set of tuples of tables: {(table1, table2), (table1, table3), ...}
    relations_corrected = {}

    for foreign_key, primary_key in relations:
        foreign_table = re.split('.csv_', foreign_key)
        primary_table = re.split('.csv_', primary_key)
        table_tuple = (foreign_table, primary_table)
        if (table_tuple in table_tuple_set):
            continue
        else:
            table_tuple_set.add(table_tuple)
            relations_corrected.add((foreign_key, primary_key))

    return relations_corrected



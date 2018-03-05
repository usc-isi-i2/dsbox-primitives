import re

class Aggregator(object):
    """
    procedure:
    1. starting from the master table. do a forward(), which find all the outcoming tables, ready to do step 3.
    2. all the forwarded tables, do a backward(), which aggreates all incoming tables, using groupby count()
    3. master table joins all the aggreated tables.

    naming conventions:
    1. for step 2, the resulted columns are renamed as: belonged_table_name+"_"+column_name+"_COUNT"
    2. for step 1, the resulted columns (joined back to master table) are renamed as: table_name+"_"+column_name
    """

    def __init__(self, relations, data, names, verbose):
        self.visited = set() # set of table names, store the tables that already in the queue (to be processed)
        self.delimiter = ".csv_"
        self.relations = relations
        self.tables = {}  # dict, key: name; value : pandas.DataFrame
        self.verbose = verbose
        counter = 0
        for x in names:
            self.tables[x] = data[counter]
            counter+=1

    def get_names(self, tableCol):
        """
        Input:
            String: eg. `loan.csv_account_id`
        Output:
            tuple of String: eg. (`loan.csv`, `account_id`)
        """
        split = re.split(self.delimiter, tableCol)
        table_name = split[0] + self.delimiter[:-1]
        column_name = split[1]
        return table_name, column_name


    def forward(self, curt_table):
        """
        Input:
            curt_table: String, name of table. eg. `loan.csv_account_id`
        Output:
            result: Pandas.DataFrame, big featurized table (join)
        """
        table_name, key_column_name = self.get_names(curt_table)
        k_tables = self.get_forward_tables(table_name)
        if self.verbose > 0: print ("current forward tables: {}".format(k_tables))
        result = self.tables[table_name]
        
        table_name_set = {}  # prevent: same table (name) happen; key is table name; value if the number of occurence

        for table_key in k_tables:
            if self.verbose > 0: print ("current forward table: {}".format(table_key))
            foreign_table_name = re.split(self.delimiter, table_key)[0] + self.delimiter[:-1]
            if (foreign_table_name in table_name_set.keys()):
                table_name_set[foreign_table_name] += 1
                foreign_table_name += str(table_name_set[foreign_table_name])
            else:
                table_name_set[foreign_table_name] = 0

            foreign_table_key = re.split(self.delimiter, table_key)[1]
            table = self.backward(table_key)
            if self.verbose > 0: print ("backward finished")
            table = table.rename(columns = lambda x : foreign_table_name+"_"+x)
            ## DEBUG code: check the intermediate tables 
            if self.verbose > 0: table.to_csv("./backwarded_table_"+foreign_table_name, index=False)

            # join back to central table, need to find the corresponding column name
            central_table_key = self.get_corresponding_column_name(table_name, table_key)
            if self.verbose > 0: print("central_table_key is: {}".format(central_table_key)) # name of primary-foreign key 
            if self.verbose > 0: print("foreign_table_key is: {}".format(foreign_table_key))
            result = result.join(other=table.set_index(foreign_table_name+"_"+foreign_table_key), 
                on=central_table_key, rsuffix="_COPY")

        return result


    def backward(self, curt_table):
        """
        Input:
            curt_table: String, name of table. eg. `account.csv_account_id`
        Output:
            result: Pandas.DataFrame, big featurized table (join of groupby count)
        """
        central_table_name, column_name = self.get_names(curt_table)
        k_tables = self.get_backward_tables(central_table_name)
        result = self.tables[central_table_name]
        if self.verbose > 0: print ("backward tables: {}".format(k_tables))
        
        for table_key in k_tables:
            # aggregated result of : groupby + count()
            if self.verbose > 0: print ("current backward table: {}".format(table_key))
            table_name, column_name = self.get_names(table_key)
            table = self.tables[table_name]
            r = table.groupby(column_name).count()      # after groupby count(), the index of r will be `column_name` automatically
            r = r.rename(columns = lambda x : table_name+"_"+x+"_COUNT")
            
            # join back to central table, need to find the corresponding column name
            central_table_key = self.get_corresponding_column_name(central_table_name, table_key)
            if self.verbose > 0: print("central_table_key is: {}".format(central_table_key)) # name of primary-foreign key 
            result = result.join(other=r, on=central_table_key) # no need to set_index for r, because its index is alreayd column_name
            
        return result

    def get_corresponding_column_name(self, table1, table2_col):
        """
        Input:
            - table1: a table name
            - table2: a table name + key column name
        Output:
            - corresponding (primary or foreign) key column name of table1
        """
        # print("DEBUG: trying to get the key for {}, with relations of {}".format(table1, table2_col))
        column_name = None
        for relation in self.relations:
            if (table2_col==relation[0]):
                if self.verbose > 0: print(relation)
                table_name = re.split(self.delimiter, relation[1])[0] + ".csv"
                if (table_name == table1):
                    column_name = re.split(self.delimiter, relation[1])[1]
                    return column_name

            elif (table2_col==relation[1]):
                if self.verbose > 0: print(relation)
                table_name = re.split(self.delimiter, relation[0])[0] + ".csv"
                if (table_name == table1):
                    column_name = re.split(self.delimiter, relation[0])[1]
                    return column_name

        raise ValueError("there is no relations between {} and {}".format(table2_col, table1))

        

            
    def get_forward_tables(self, table_name):
        """
        Input:
            table_name: string, eg: "loan.csv"

        Output:
            list of String, String is the second element in `relations` tuples (Primary key),
            which is the forward table of current table
        """
        result = list()
        for relation in self.relations:
            # assumption: the name like "loan.csv", always represent a table
            if (table_name in relation[0]): 
                result.append(relation[1])
                self.visited.add(relation[0])
        return result

    def get_backward_tables(self, table_name):
        """
        Output:
            list of String, String is the first element in `relations` tuples (Primary key)
        """
        result = list()
    #     print (visited)
        for relation in self.relations:
            if (table_name in relation[1] and (relation[0] not in self.visited)):
                result.append(relation[0])
    #             visited.add(relation[0])
        return result
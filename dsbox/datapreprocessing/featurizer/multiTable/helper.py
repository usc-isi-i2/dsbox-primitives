import re

class Aggregator(object):


    def __init__(self, relations, data, names):
        self.visited = set() # set of table names, store the tables that already in the queue (to be processed)
        self.delimiter = ".csv_"
        self.relations = relations
        self.tables = {}  # dict, key: name; value : pandas.DataFrame
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
        print (k_tables)
        result = self.tables[table_name]
        
        for table in k_tables:
            foreign_table_name = re.split(self.delimiter, table)[0] + self.delimiter[:-1]
            table = self.backward(table)
            print(key_column_name) # name of primary-foreign key 
            result = result.join(table.set_index(key_column_name), on=key_column_name, lsuffix="_"+table_name, rsuffix="_"+foreign_table_name)

        return result


    def backward(self, curt_table):
        """
        Input:
            curt_table: String, name of table. eg. `account.csv_account_id`
        Output:
            result: Pandas.DataFrame, big featurized table (join of groupby count)
        """
        table_name, column_name = self.get_names(curt_table)
        k_tables = self.get_backward_tables(table_name)
        result = self.tables[table_name]
        
        for table in k_tables:
            # aggregated result of : groupby + count()
            table_name, column_name = self.get_names(table)
            table = self.tables[table_name]
            r = table.groupby(column_name).count()
            r = r.rename(columns = lambda x : table_name+"_"+x)
            
            result = result.join(r, on=column_name)
            
        return result
            
    def get_forward_tables(self, table_name):
        """
        Output:
            list of String, String is the second element in `relations` tuples (Primary key)
        """
        result = list()
        for relation in self.relations:
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
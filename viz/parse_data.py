import os
import csv

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class DataParser():
    def __init__(self, files_to_load=["test_512", "test_1024", "test_2048", "test_4096"],
        replace_dict={}, filter_col_list=[]):
        self.files_to_load = files_to_load
        self.replace_dict = replace_dict
        self.filter_col_set = set(filter_col_list)
        self.csv_path = os.path.join(BASE_PATH, "..", "data")
        self.header = None
        self.row_dicts = []
        self.parse_results()
    
    def parse_results(self):
        """Sets self.row_dicts

        Converts every row into a dictionary representation of the row
        """
        # Iterate through every csv file in data
        for file_name in self.files_to_load:
            csv_header = None
            with open(os.path.join(self.csv_path, file_name + ".csv"), "r") as csv_file:
                csv_reader = csv.reader(csv_file)

                # Get header and set to the object if not yet seen
                if csv_header is None:
                    csv_header = next(csv_reader)
                else:
                    next(csv_reader)
                
                for row in csv_reader:
                    row_dict = {}

                    for col_index, value in enumerate(row):
                        row_dict[csv_header[col_index]] = value
                    
                    if 'node_count' not in row_dict:
                        row_dict['node_count'] = file_name.split("_")[1]
                    
                    if 'bw_level' not in row_dict:
                        row_dict['bw_level'] = 3 + int(row_dict['ipath_1']) - int(row_dict['ipath_0'])
                    
                    new_row_dict = {}
                    for key, val in row_dict.items():
                        if key not in self.filter_col_set:
                            if key in self.replace_dict:
                                new_row_dict[self.replace_dict[key]] = val
                            else:
                                new_row_dict[key] = val
                    
                    if self.header is None:
                        self.header = sorted(new_row_dict.keys())
                    
                    row_value_list = []
                    for val in self.header:
                        row_value_list.append(new_row_dict[val])

                    for key, val in new_row_dict.items():
                        if key == "bw_level": continue

                        if val.isdigit():
                            new_row_dict[key] = int(val)
                        # Convert to float if applicable
                        elif val.replace('.', '').isdigit():
                            new_row_dict[key] = float(val)

                    new_row_dict['row'] = row_value_list

                    self.row_dicts.append(new_row_dict)

    
    def split(self, split_on_category='all', prefix='', suffix='', lambda_filter=None, safe_mode=True):
        """Writes out several csv files depending on parameters

        split_on_category - (String) A string whose value must be either 'all' or
            a value in the header of the original csv files, e.g., 'app_0'. The method
            will make a csv file for each unique occurence of a value under the split_on_category
        suffix - (String) A string to append onto each csv file created, e.g., '_abc' will cause
            csv1_abc.csv, csv2_abc.csv, ... and so on
        lambda_filter (lambda function) A lambda function that will be given a row dictionary as its
            argument and is expected to return True or False. Only the row dictionaries that produce
            a True value will be included in the final csv files. e.g.,
            'lambda_filter=lambda x: x[test_id]==1024' will only include rows whose test_id column is 1024
        safe_mode (Boolean) If more than 10 csv files will be produce, the program will by default not allow
            that to happen. Setting safe_mode=False will bypass this safety check
        """
        # Create a copy of the row dictionaries
        row_dict_list = self.row_dicts.copy()

        # If lambda_filter was given, filter based on the lambda function
        if lambda_filter is not None:
            row_dict_list = self.filter_on_lambda(row_dict_list, lambda_filter)

        # Split based on the category name
        # Expect form of [(x1, y1), (x2, y2), ...]
        # Where x1 is a csv file path and y1 is a list of rows
        csv_list = self.split_category(row_dict_list, split_on_category, prefix, suffix, safe_mode)

        # If none too many csv files were requested
        if csv_list is None:
            return

        # Write out new csv files
        for csv_path, rows in csv_list:
            with open(csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(self.header)

                for row in rows:
                    csv_writer.writerow(row)
    
    def split_category(self, row_dict_list, split_on_category, prefix, suffix, safe_mode):
        """Returns a list of tuples of csv file paths and their respective rows

        row_dict_list - (List of Dictionary) A list of dictionaries representing rows
        split_on_category - See def split
        suffix - See def split
        safe_mode - See def split
        """

        # Handles situation of creating only one file
        if split_on_category == 'all':
            file_name = prefix + "all" + suffix + ".csv"
            file_path = os.path.join(self.csv_path, file_name)

            row_list = []
            for row_dict in row_dict_list:
                row_list.append(row_dict['row'])

            return [(file_path, row_list)]
    
        # Iterate over each row dictionary and create a new dictionary whose 
        # Keys are the categories and whose values are rows who fit that category
        categories = {}
        for row_dict in row_dict_list:
            key = str(row_dict[split_on_category])
            if key not in categories:
                categories[key] = [row_dict['row']]
            else:
                categories[key].append(row_dict['row'])
        
        # Safety check to assure not no more than 10 files are produced unless the
        # user specifically requests so
        if safe_mode and len(categories) > 10:
            print("WARNING: Splitting on a category that will produce %s files. Set safe_mode=False to continue..." % len(categories))
            return None
        
        # Create csv paths and tuples
        return_list = []
        for category, row_list in categories.items():
            file_name = prefix + category.lower() + suffix + ".csv"
            file_path = os.path.join(self.csv_path, file_name)
            return_list.append((file_path, row_list))

        return return_list


    def filter_on_lambda(self, row_dict_list, lambda_filter):
        """Handles filtering a list of dictionaries based on a lambda function

        row_dict_list - (List of Dictionary) A list of dictionaries representing rows
        lambda_filter - See def split
        """
        new_row_dict_list = []

        for row_dict in row_dict_list:
            # If it passes the lambda check, include it
            if lambda_filter(row_dict):
                new_row_dict_list.append(row_dict)

        return new_row_dict_list


def main():
    # Base filter settings
    replace_dict = {
        "power_limit_0": "power_cap",
        "app_0": "app",
        "thread_count_0": "thread_count",
        "algorithm_0": "algorithm",
        "runtime_0": "runtime"
    }
    filter_col_list = ["run_no", "stub", "start_0", "end_0", "srun_time_0",
        "thread_count_1", "algorithm_1", "start_1", "int_iterations", "int_avg_latency",
        "int_min_latency", "int_max_latency", "app_1", "power_limit_1"]

    data_parser = DataParser(replace_dict=replace_dict, filter_col_list=filter_col_list)

    # data_parser.split(split_on_category='app')

    # Example of unioning all the csv files 
    #data_parser.split(split_on_category='all')
    
    # Example of splitting on test_count (512, 1024, ...)
    #data_parser.split(split_on_category='test_count')
    
    # Example of creating one file but ONLY including rows whose test_count is 512 or 1024
    #data_parser.split(split_on_category='all', suffix='_512_1024', lambda_filter=lambda x: x['test_count'] == 512 or x['test_count'] == 1024)
    
    # Example of creating multiple files based on the value of app_0
    #data_parser.split(split_on_category='app_0')
    
    # Example of splitting on app_0 AND filtering results based on test_count
    #data_parser.split(split_on_category='app_0', suffix='_512_1024', lambda_filter=lambda x: x['test_count'] == 512 or x['test_count'] == 1024)
    
    # Example of a split that will produce a LOT of files. Safety check triggers
    #data_parser.split(split_on_category='runtime_0')

if __name__ == "__main__":
    main()

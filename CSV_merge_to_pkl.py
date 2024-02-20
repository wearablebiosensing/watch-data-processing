import sys
import csv
from datetime import datetime
import pandas as pd
import json
import glob
import os
import calendar
import joblib
import numpy as np 
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pickle

def parse_timestamp(timestamp_str):
    # Parse the timestamp string and extract the time part
    timestamp_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
    return timestamp_obj

def clear_output_file(output_file):
    with open(output_file, 'w', newline=''):
        pass

def create_dataframes(input_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        
        # Read the header
        header = next(reader)

        # Initialize variables
        dataframes_list = []
        current_dataframe_rows = []

        for line in reader:
            # Replace empty cells or cells containing spaces with NaN
            line = [cell if cell.strip() else np.nan for cell in line]

            if not any(pd.isna(cell) for cell in line):
                # Empty line encountered, create a new DataFrame
                if current_dataframe_rows:
                    current_dataframe = pd.DataFrame(current_dataframe_rows, columns=header)
                    
                    # Convert all columns (excluding timestamp columns) to floats
                    non_timestamp_columns = current_dataframe.columns.difference(['Accel_Timestamp', 'Gyro_Timestamp','Accel_event.timestamp', 'Gyro_event.timestamp'])
                    current_dataframe[non_timestamp_columns] = current_dataframe[non_timestamp_columns].astype(float)
                    
                    dataframes_list.append(current_dataframe)
                    current_dataframe_rows = []  # Reset for the next set of lines
            else:
                # Accumulate lines for the current set
                current_dataframe_rows.append(line)

        # Create a DataFrame for the last set of lines
        if current_dataframe_rows:
            current_dataframe = pd.DataFrame(current_dataframe_rows, columns=header)
            
            # Convert all columns (excluding timestamp columns) to floats
            non_timestamp_columns = current_dataframe.columns.difference(['Accel_Timestamp', 'Gyro_Timestamp','Accel_event.timestamp', 'Gyro_event.timestamp'])
            current_dataframe[non_timestamp_columns] = current_dataframe[non_timestamp_columns].astype(float)
            
            dataframes_list.append(current_dataframe)

    return dataframes_list

# features to analyse
# mean
def feature_mean(feature_name,accel_gyro_data):
    mean = accel_gyro_data.mean()
    mean_transposed_df = mean.to_frame().T
    mean_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name]
    return mean_transposed_df

#std
def feature_std(feature_name,accel_gyro_data):
    std = accel_gyro_data.std()
    std_transposed_df = std.to_frame().T
    std_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" +feature_name]
    return std_transposed_df

#kurtosis
def feature_kurtosis(feature_name,accel_gyro_data):
    kurtosis = accel_gyro_data.kurtosis()
    kurtosis_transposed_df = kurtosis.to_frame().T
    kurtosis_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_"+feature_name]
    return kurtosis_transposed_df

#median
def feature_median(feature_name,accel_gyro_data):
    median = accel_gyro_data.median()
    median_transposed_df = median.to_frame().T
    median_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" + feature_name]
    return median_transposed_df

#skewness
def feature_skewness(feature_name,accel_gyro_data):
    skewness = accel_gyro_data.skew()
    skewness_transposed_df = skewness.to_frame().T
    skewness_transposed_df.columns = ["Accel_X_"+feature_name ,"Accel_Y_" +feature_name ,"Accel_Z_" +feature_name,"Gyro_X_" +feature_name, "Gyro_Y_" + feature_name ,"Gyro_Z_" + feature_name]
    return skewness_transposed_df


# 'Accel_X_mean', 'Accel_Y_mean', 'Accel_Z_mean', 'Gyro_X_mean',
#        'Gyro_Y_mean', 'Gyro_Z_mean', 'Accel_X_std', 'Accel_Y_std',
#        'Accel_Z_std', 'Gyro_X_std', 'Gyro_Y_std', 'Gyro_Z_std',
#        'Accel_X_kurtosis', 'Accel_Y_kurtosis', 'Accel_Z_kurtosis',
#        'Gyro_X_kurtosis', 'Gyro_Y_kurtosis', 'Gyro_Z_kurtosis',
#        'Accel_X_skewness', 'Accel_Y_skewness', 'Accel_Z_skewness',
#        'Gyro_X_skewness', 'Gyro_Y_skewness', 'Gyro_Z_skewness', 'type'],
# create table 
def get_concated_features(accel_gyro_data):
    # Subset the DataFrame to include only the specified columns
    accel_gyro_data = accel_gyro_data[['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
    df1 = feature_mean("mean",accel_gyro_data)
    # df2 = feature_median("median",accel_gyro_data)
    df3 = feature_std("std",accel_gyro_data)
    df4 = feature_kurtosis("kurtosis",accel_gyro_data)
    df5 =feature_skewness("skewness",accel_gyro_data)
    df = [df1, df3, df4, df5]
    result_df = pd.concat(df, axis=1)

    return result_df

def merge_and_save(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f1, open(input_file2, 'r') as f2, open(output_file, 'w', newline='') as output:
        # Initialize CSV reader and writer
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(output)

        # Read headers from both files
        header1 = next(reader1)
        header2 = next(reader2)

        # Rename categories in the second file
        header1 = ['Accel_' + category for category in header1]

        # Rename categories in the second file
        header2 = ['Gyro_' + category for category in header2]

        # Write the merged headers to the output file
        writer.writerow(header1 + header2)

        # Initialize the second variable
        second = None

        # Read the first lines from both files
        line1 = next(reader1, None)
        line2 = next(reader2, None)

        while line1 is not None and line2 is not None:
            timestamp1 = parse_timestamp(line1[-1])
            timestamp2 = parse_timestamp(line2[-1])

            if abs((timestamp1 - timestamp2).total_seconds()) <= 1:
                if timestamp1.second == timestamp2.second:
                    # Combine lines and push to the output file
                    combined_line = line1 + line2
                    writer.writerow(combined_line)
                else:
                    # Add an empty line between seconds
                    writer.writerow([])

            # Determine which file to read the next line from based on the timestamp
            if timestamp1 <= timestamp2:
                line1 = next(reader1, None)
            else:
                line2 = next(reader2, None)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python merge_and_trim_lines.py <input_file1> <input_file2> <output_file>")
        sys.exit(1)

    # Extract input file paths and output file path from command-line arguments
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    output_file = sys.argv[3]

    # Call the function to merge and trim lines from both files
    merge_and_save(input_file1, input_file2, output_file)
    dfList = create_dataframes(output_file)
    clear_output_file(output_file)
    with open('AD-155_Deployment.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    # print(dfList)
    results = []
    for i in dfList:
        results.append(loaded_model.predict(get_concated_features(i)))
        # results.append(get_concated_features(i))
    
    print([j.item() for j in results])
    # for idx, result_df in enumerate(results):
    #     print(f"DataFrame {idx + 1}:")
    #     print(result_df)
    #     print("\n" + "=" * 50 + "\n")  # Separator between dataframes for better readability
    # Load the pickled object (e.g., a trained machine learning model)
    # print(f"DataFrame {1}:")
    # print(results[0].dtypes)
    # print("\n" + "=" * 50 + "\n")

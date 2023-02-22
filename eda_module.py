# python libs
import os
import glob
from collections import OrderedDict
import re

# data manipulation libs
import numpy as np
import pandas as pd

# data viz libs
# https://pandas.pydata.org/pandas-docs/version/1.0/user_guide/style.html
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.figsize": (12, 8)})

# format list to tabular form
from tabulate import tabulate

# no. of rows to display; passing None displays all rows
pd.set_option('display.max_rows', 100)

# no. of cols to display; passing None displays all rows
pd.set_option("display.max_columns", None)

# float show comma separators
pd.options.display.float_format = '{:_.2f}'.format
# alternative:
# df.head().style.format("{:,.0f}")
# df.head().style.format({"col1": "{:,.0f}", "col2": "{:,.0f}"})
# more here: https://pbpython.com/styling-pandas.html

# to format integers in a dataframe use the following method
#df.style.format(thousands=',')

import warnings

warnings.filterwarnings("ignore")


def table(df: pd.DataFrame) -> None:
    """Print basic dataframe stats in a tabular form. General EDA function to get a first overview and sample of the data frame

    Args:
        df (pd.DataFrame): Dataframe of interest

    Returns:
        None
    """
    rows = []  # initialize an empty list to store rows
    row_no = 1  # initialize the row number
    max_list_len = 8  # maximum length of a list to be displayed in the unique values column
    max_concat_list_len = 75 # maximum length of a concatenated list to be displayed in the unique values column

    # Loop through each column in the dataframe and create a row for the table
    for row_no, col in enumerate(df):
        # Assign the row number, column name, and dtype
        row = [row_no, col, str(df[col].dtype)]

        # Depending on the data type and number of unique values in the column, extend the row with either:
        #   - the number of unique values (if the column is an array)
        #   - the number of unique values (if the number of unique values is above the threshold)
        #   - the unique values themselves (if the number of unique values is below the threshold)
        if type(df[col].iloc[0]) == np.ndarray:
            # old version of the function
            # col_transformed = pd.Series([','.join(map(str, l)) for l in df[col]])
            col_transformed = pd.Series([','.join(map(str, l)) for l in df[col]]).sort_values()  # convert array values to a string with elements separated by commas
            row.extend([f'{col_transformed.nunique():_}'])  # add the number of unique values to the row
            row.extend([
                f'{col_transformed.isna().sum():_}',  # add the number of NAs in the column to the row
                f'{len(df) - np.count_nonzero(col_transformed):_}'  # add the number of zeros and falses in the column to the row
            ])
        elif df[col].nunique() > max_list_len:
            row.extend([f'{df[col].nunique():_}'])  # add the number of unique values to the row
            row.extend([
                f'{df[col].isna().sum():_}',  # add the number of NAs in the column to the row
                f'{len(df) - np.count_nonzero(df[col]):_}'  # add the number of zeros and falses in the column to the row
            ])
        else:
            # old version of the function
            # unique_values = sorted(list(df[col].unique())) # sort the unique values
            # row.append(unique_values)   # add the list of unique values to the row
            unique_values = sorted(list(df[col].unique())) # sort the unique values
            unique_values_concat = ', '.join(map(str, unique_values))  # concatenate the unique values into a string
            if len(unique_values_concat) > max_concat_list_len:
                unique_values_concat = f"{unique_values_concat[:max_concat_list_len-3]}..."  # add three dots if the concatenated values exceed the threshold
            row.append(unique_values_concat)  # add the list of unique values to the row
            row.extend([
                f'{df[col].isna().sum():_}',  # add the number of NAs in the column to the row
                f'{len(df) - np.count_nonzero(df[col]):_}'  # add the number of zeros and falses in the column to the row
            ])
        # Append the row to the rows list
        rows.append(row)

    # Create and print table using the tabulate library
    table = tabulate(
        rows,
        headers=["n", "col_name", "dtype","unique_values", "NAs", "0s/Fs"],
        tablefmt="pipe")
    print(f"Number of records: {len(df):_}\n")
    print(table)
    return df.sample(5)  # return a sample of the dataframe

def list_to_string(main_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert a list column to string in a Pandas DataFrame

    Args:
        main_df (pd.DataFrame): The input DataFrame
        cols (list): The list of column names to convert to string

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns converted to string
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = main_df.copy()

    # Iterate over each column and convert it to a string
    for col in cols:
        df[col] = pd.Series([','.join(map(str, l)) for l in df[col]])
        
    return df

def all_lists_to_string(main_df: pd.DataFrame) -> pd.DataFrame:
    """Convert all list columns to string in a Pandas DataFrame

    Args:
        main_df (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: A new DataFrame with all list columns converted to string
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = main_df.copy()

    # Iterate over each column and convert it to a string if it's a list or ndarray
    for col in df.columns:
        if isinstance(df[col].iloc[0], list) or isinstance(df[col].iloc[0], np.ndarray):
            df[col] = pd.Series([', '.join(map(str, l)) for l in df[col]])

    return df


def flatten_multiindex(df: pd.DataFrame) -> list:
    """Flatten and reverse multiindex columns

    Args:
        df (pd.DataFrame): The input DataFrame with multi-index columns

    Returns:
        list: A list of column names with flattened multi-index
    """
    # Combine the first and second level column names into a single string with an underscore separator
    cols = ['_'.join(col).strip('_') for col in df.columns.values]
    
    # Return the list of column names
    return cols


def wavg(df: pd.DataFrame, weight: str, value: str) -> float:
    """
    Calculate the weighted average of a column in a DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        weight (str): The name of the column containing the weights.
        value (str): The name of the column containing the values.
    
    Returns:
        float: The weighted average of the values in the specified column.
    """
    # Calculate the weighted column
    weighted_col = df[weight] * df[value]
    
    # Calculate the sum of the weighted column and the sum of the weights
    wsum = df[weight].sum()
    if wsum == 0:
        raise ZeroDivisionError("The sum of weights is zero.")
    else:
        # Calculate the weighted average
        wavg = weighted_col.sum() / wsum
    
    return wavg

# Useful snippets

# Create dict to map values based on an Excel table
# mm = pd.read_clipboard().dropna()
# mapping = mm.set_index('col_name')['rename_to'].to_dict()

# Reorder cols based on an Excel col
# mm = pd.read_clipboard().dropna()
# oo = mm.iloc[:, 0].tolist()
# df = df[oo]
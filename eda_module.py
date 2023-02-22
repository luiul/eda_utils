# In case chatPGT does not finish code suggestion see: https://www.reddit.com/r/OpenAI/comments/zgkulg/chatgpt_often_will_not_finish_its_code_or/
# Examples for prompts to finish code suggestion: 
# Finish your answer
# Continue from the last line
# Print the rest of the code without reprinting what you've just showed me
# Finish the code. Do not print the full code again, just a missing part from last answer

# python libs
import os 
import glob 
from collections import OrderedDict 
import re 
from typing import List, Tuple # for type hinting
import warnings # to ignore (some) warnings

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

warnings.filterwarnings("ignore")

def table(df: pd.DataFrame) -> pd.DataFrame:
    """Print basic dataframe stats in a tabular form. General EDA function to get a first overview and sample of the data frame

    Args:
        df (pd.DataFrame): Dataframe of interest

    Returns:
        pd.DataFrame: A sample of the dataframe
    """
    rows: List[List] = []  # initialize an empty list to store rows
    row_no: int = 1  # initialize the row number
    max_list_len: int = 8  # maximum length of a list to be displayed in the unique values column
    max_concat_list_len: int = 75 # maximum length of a concatenated list to be displayed in the unique values column

    # Loop through each column in the dataframe and create a row for the table
    for row_no, col in enumerate(df):
        # Assign the row number, column name, and dtype
        row: List = [row_no, col, str(df[col].dtype)]

        # Depending on the data type and number of unique values in the column, extend the row with either:
        #   - the number of unique values (if the column is an array)
        #   - the number of unique values (if the number of unique values is above the threshold)
        #   - the unique values themselves (if the number of unique values is below the threshold)
        if type(df[col].iloc[0]) == np.ndarray:
            col_transformed: pd.Series = pd.Series([','.join(map(str, l)) for l in df[col]]).sort_values()  # convert array values to a string with elements separated by commas
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
            unique_values: List = sorted(list(df[col].unique())) # sort the unique values
            unique_values_concat: str = ', '.join(map(str, unique_values))  # concatenate the unique values into a string
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
    table: str = tabulate(
        rows,
        headers=["n", "col_name", "dtype","unique_values", "NAs", "0s/Fs"],
        tablefmt="pipe")
    print(f"Number of records: {len(df):_}\n")
    print(table)
    return df.sample(5)  # return a sample of the dataframe

def list_to_string(main_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert a list column to string in a Pandas DataFrame

    Args:
        main_df (pd.DataFrame): The input DataFrame
        cols (List[str]): The list of column names to convert to string

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns converted to string
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df: pd.DataFrame = main_df.copy()

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
    df: pd.DataFrame = main_df.copy()

    # Iterate over each column and convert it to a string if it's a list or ndarray
    for col in df.columns:
        if isinstance(df[col].iloc[0], list) or isinstance(df[col].iloc[0], np.ndarray):
            df[col] = pd.Series([', '.join(map(str, l)) for l in df[col]])

    return df


def flatten_multiindex(df: pd.DataFrame) -> List[str]:
    """Flatten and reverse multiindex columns

    Args:
        df (pd.DataFrame): The input DataFrame with multi-index columns

    Returns:
        List[str]: A list of column names with flattened multi-index
    """
    # Combine the first and second level column names into a single string with an underscore separator
    cols: List[str] = ['_'.join(col).strip('_') for col in df.columns.values]
    
    # Return the list of column names
    return cols


def weighted_avg(group: pd.DataFrame) -> float:
    """Calculate weighted average of values in a group of rows.

    Args:
        group (pd.DataFrame): Group of rows to calculate weighted average for.

    Returns:
        float: Weighted average of values in the group.
    """
    w: pd.Series = group['weight_col']
    v: pd.Series = group['val_col']
    total_weight: float = w.sum()
    if total_weight == 0:
        raise ValueError("Sum of weights in group is zero")
    return (w * v).sum() / total_weight

def group_weighted_avg(
    df: pd.DataFrame, group_col: str, val_col: str, weight_col: str
) -> pd.Series:
    """Group a Pandas DataFrame by a column and calculate the weighted average of another column.

    Args:
        df (pd.DataFrame): DataFrame to group and calculate weighted average for.
        group_col (str): Name of the column to group the DataFrame by.
        val_col (str): Name of the column to calculate the weighted average of.
        weight_col (str): Name of the column to use as weights in the weighted average calculation.

    Returns:
        pd.Series: Series containing the weighted average of values in `val_col` for each group in the DataFrame,
            indexed by the unique values in the `group_col` column.
    """
    grouped: pd.DataFrame = df.groupby(group_col).apply(weighted_avg)
    return grouped
# Example usage
# df: pd.DataFrame = pd.DataFrame({
#     'group_col': ['A', 'A', 'B', 'B'],
#     'val_col': [1, 2, 3, 4],
#     'weight_col': [0.1, 0.2, 0.3, 0.4]
# })
# grouped: pd.Series = group_weighted_avg(df, 'group_col', 'val_col', 'weight_col')
# print(grouped)



# Other useful snippets

# Create dict to map values based on an Excel table
# mm = pd.read_clipboard().dropna()
# mapping = mm.set_index('col_name')['rename_to'].to_dict()

# Reorder cols based on an Excel col
# mm = pd.read_clipboard().dropna()
# oo = mm.iloc[:, 0].tolist()
# df = df[oo]
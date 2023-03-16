# python libs
import os
import glob
from collections import OrderedDict
import re
from typing import List, Tuple, Union, Optional
import warnings  # to ignore (some) warnings

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
    max_list_len: int = 10  # maximum length of a list to be displayed in the unique values column
    max_concat_list_len: int = 75  # maximum length of a concatenated list to be displayed in the unique values column

    # Loop through each column in the dataframe and create a row for the table
    for row_no, col in enumerate(df):
        # Assign the row number, column name, and dtype
        row: List = [row_no, col, str(df[col].dtype)]

        # Depending on the data type and number of unique values in the column, extend the row with either:
        #   - the number of unique values (if the column is an array)
        #   - the number of unique values (if the number of unique values is above the threshold)
        #   - the unique values themselves (if the number of unique values is below the threshold)
        if type(df[col].iloc[0]) == np.ndarray:
            col_transformed: pd.Series = pd.Series([
                ','.join(map(str, l)) for l in df[col]
            ]).sort_values(
            )  # convert array values to a string with elements separated by commas
            row.extend([f'{col_transformed.nunique():_}'
                        ])  # add the number of unique values to the row
            row.extend([
                f'{col_transformed.isna().sum():_}',  # add the number of NAs in the column to the row
                f'{len(df) - np.count_nonzero(col_transformed):_}'  # add the number of zeros and falses in the column to the row
            ])
        elif df[col].nunique() > max_list_len:
            row.extend([f'{df[col].nunique():_}'
                        ])  # add the number of unique values to the row
            row.extend([
                f'{df[col].isna().sum():_}',  # add the number of NAs in the column to the row
                f'{len(df) - np.count_nonzero(df[col]):_}'  # add the number of zeros and falses in the column to the row
            ])
        else:
            # unique_values: List = sorted(list(df[col].unique()))  # sort the unique values
            unique_values: List = sorted([
                str(val) for val in df[col].unique()
            ])  # cast to string before sorting (otherwise comparisson fails)
            unique_values_concat: str = ', '.join(map(
                str,
                unique_values))  # concatenate the unique values into a string
            if len(unique_values_concat) > max_concat_list_len:
                unique_values_concat = f"{unique_values_concat[:max_concat_list_len-3]}..."  # add three dots if the concatenated values exceed the threshold
            row.append(unique_values_concat
                       )  # add the list of unique values to the row
            row.extend([
                f'{df[col].isna().sum():_}',  # add the number of NAs in the column to the row
                f'{len(df) - np.count_nonzero(df[col]):_}'  # add the number of zeros and falses in the column to the row
            ])
        # Append the row to the rows list
        rows.append(row)

    # Create and print table using the tabulate library
    table: str = tabulate(
        rows,
        headers=["n", "col_name", "dtype", "unique_values", "NAs", "0s/Fs"],
        tablefmt="pipe")
    print(f"Number of records: {len(df):_}\n")
    print(table)
    if len(df) > 5:
        return df.sample(5)  # return a sample of the dataframe
    else:
        return df
    # return df.sample(5)  # return a sample of the dataframe


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
        if isinstance(df[col].iloc[0], list) or isinstance(
                df[col].iloc[0], np.ndarray):
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


def weighted_avg(group: pd.DataFrame, weight_col: str, val_col: str) -> float:
    """Calculate weighted average of values in a group of rows.

    Args:
        group (pd.DataFrame): Group of rows to calculate weighted average for.
        weight_col (str): Name of the column to use as weights in the weighted average calculation.
        val_col (str): Name of the column to calculate the weighted average of.

    Returns:
        float: Weighted average of values in the group.
    """
    w: pd.Series = group[weight_col]
    v: pd.Series = group[val_col]
    total_weight: float = w.sum()
    if total_weight == 0:
        raise ZeroDivisionError("The sum of weights is zero.")
    return (w * v).sum() / total_weight


def group_weighted_avg(df: pd.DataFrame, group_col: Union[str, List[str]],
                       val_col: str, weight_col: str,
                       col_name: str) -> pd.DataFrame:
    """Group a Pandas DataFrame by a column or a list of columns and calculate the weighted average of another column.

    Args:
        df (pd.DataFrame): DataFrame to group and calculate weighted average for.
        group_col (Union[str, List[str]]): Name or list of the column(s) to group the DataFrame by.
        val_col (str): Name of the column to calculate the weighted average of.
        weight_col (str): Name of the column to use as weights in the weighted average calculation.
        col_name (str): Name of the resulting column.

    Returns:
        pd.DataFrame: DataFrame containing the weighted average of values in `val_col` for each group in the DataFrame,
            indexed by the unique values in the `group_col` column.
    """
    grouped: pd.DataFrame = df.groupby(group_col, as_index=False).apply(
        weighted_avg, weight_col, val_col)
    grouped.columns = grouped.columns.fillna(col_name)
    return grouped


def group_merge_weighted_avg(
        df: pd.DataFrame,
        group_col: Union[str, List[str]],
        val_col: str,
        weight_col: str,
        col_name: str,
        merge_col: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
    """Group a Pandas DataFrame by a column or a list of columns, calculate the weighted average of another column, and merge the results back to the original DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to group and calculate weighted average for.
        group_col (Union[str, List[str]]): Name or list of the column(s) to group the DataFrame by.
        val_col (str): Name of the column to calculate the weighted average of.
        weight_col (str): Name of the column to use as weights in the weighted average calculation.
        col_name (str): Name of the resulting column.
        merge_col (Union[str, List[str]], optional): Name or list of the column(s) to merge the resulting DataFrame on. If None, defaults to group_col. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the original columns and the new column with the weighted average for each group.
    """
    grouped: pd.Series = group_weighted_avg(df, group_col, val_col, weight_col,
                                            col_name)
    merge_col = merge_col or group_col
    merged: pd.DataFrame = pd.merge(df, grouped, how="left", on=merge_col)
    # merged.drop(columns=group_col, inplace=True)
    return merged


def create_store_table(file_path: str) -> pd.DataFrame:
    # If file exists, load it; otherwise, read clipboard and save to file
    if os.path.isfile(file_path):
        read_data = pd.read_csv(file_path)
    else:
        # Read clipboard and create mapping dataframe
        read_data = pd.read_clipboard()
        # Save mapping dataframe to file
        read_data.to_csv(file_path, index=False)

    # Create mapping dictionary
    table = read_data
    return table


def create_store_mapping(file_path: str) -> dict:
    """Reads, saves, or creates a mapping file and returns a dictionary of column name mappings.

    Args:
        file_path: The path to the mapping file.

    Returns:
        A dictionary of column name mappings, where the keys are the original column names and the values are the new
        column names.

    """
    # If file exists, load it; otherwise, read clipboard and save to file
    if os.path.isfile(file_path):
        read_data = pd.read_csv(file_path)
    else:
        # Read clipboard and create mapping dataframe
        read_data = pd.read_clipboard().dropna()
        # Save mapping dataframe to file
        read_data.to_csv(file_path, index=False)

    # Create mapping dictionary
    mapping = read_data.set_index(
        read_data.columns[0])[read_data.columns[1]].to_dict()
    return mapping


def create_store_col_order(file_path: str) -> list:
    """Creates a column order mapping based on an Excel table, saves it to a file, and returns a list of column names.

    If the mapping file exists at the specified file path, the function loads the mapping file and returns a list of
    column names in the order specified in the file. If the mapping file does not exist, the function reads the clipboard
    to create the mapping dataframe, saves it to file, and returns a list of column names in the order specified in the
    clipboard data.

    Args:
        file_path: The path to the mapping file.

    Returns:
        A list of column names in the order specified by the Excel table.

    """
    # If file exists, load it; otherwise, read clipboard and save to file
    if os.path.isfile(file_path):
        read_data = pd.read_csv(file_path)
    else:
        # Read clipboard and create mapping dataframe
        read_data = pd.read_clipboard().dropna()
        # Save mapping dataframe to file
        read_data.to_csv(file_path, index=False)

    # Create col order list
    col_order = read_data.iloc[:, 0].tolist()

    return col_order


def touch(my_file: str) -> str:
    """Returns the file path for a file with the specified name located in the 'data' directory of the current working
    directory.

    Args:
        my_file: A string representing the name of the file to create or retrieve the path to.

    Returns:
        A string representing the file path for a file with the specified name located in the 'data' directory of the
        current working directory.

    """
    current_directory = os.getcwd()
    data_file_path = os.path.join(current_directory, 'data', my_file)
    return data_file_path

def print_list(obj):
    """
    Given an object, check if it is a list and print each element of the list on a new line.

    Args:
        obj: An object to print. If obj is not a list, it will be cast to a list.

    Returns:
        None
    """
    if not isinstance(obj, list):
        obj = list(obj)
    for item in obj:
        print(item)



# # ------------------------------------ Other functinos ------------------------------------
# def pwd() -> str:
#     """Returns the current working directory.

#     Returns:
#         A string representing the current working directory.

#     """
#     return os.getcwd()

# # ------------------------------------ Examples ------------------------------------
# df: pd.DataFrame = pd.DataFrame({
#     'group_col': ['A', 'A', 'B', 'B'],
#     'group_col2': ['C', 'C', 'C', 'E'],
#     'values': [1, 2, 3, 4],
#     'weights': [0.1, 0.2, 0.3, 0.4]
# })

# group_test = ['group_col',['group_col','group_col2']]

# print(df)

# for test in group_test:
#     print('\n')
#     print(f'------------------ Grouping by: {test} ------------------')
#     grouped= group_weighted_avg(df, test, 'values', 'weights', 'wavg')
#     grouped_merge= group_merge_weighted_avg(df, test, 'values', 'weights', 'wavg')
#     print(grouped)
#     print(grouped_merge)

# # ------------------------------------ Useful snippets ------------------------------------
# # Determine what columns to group by
# group = 'order_number'

# # Create a new DataFrame with the unique SKU counts for each order_number
# sku_counts = df.groupby(group)['sku'].nunique().rename('sku_count').reset_index()

# # Merge the original DataFrame with the new DataFrame
# df_merged = df.merge(sku_counts, on=group)

# # Sort the merged DataFrame by the unique SKU counts
# df_sorted = df_merged.sort_values(by='sku_count', ascending=False)

# # Print the sorted DataFrame
# df_sorted

# def raw_prices_agg(df: pd.DataFrame, weight_kg: str, spend_eur: str,
#                    price_eur: str) -> pd.Series:
#     """
#     Calculates various raw price aggregates for a given input DataFrame.

#     Args:
#         df: The input DataFrame.
#         weight_kg: The name of the column used as weight in kilograms.
#         spend_eur: The name of the column used as spend in EUR.
#         price_eur: The name of the column used as price in EUR.

#     Returns:
#         A pandas Series containing the calculated raw price aggregates.
#     """
#     o_dict: OrderedDict = OrderedDict()

#     sum_volume: float = df[weight_kg].sum()

#     weight_eur: pd.Series = df[weight_kg] * df[price_eur]
#     avg_price_eur: float = weight_eur.sum() / df[weight_kg].sum()
#     sum_spend_eur: float = df[spend_eur].sum()

#     o_dict['sum_volume'] = sum_volume
#     o_dict['avg_price_eur'] = avg_price_eur
#     o_dict['sum_spend_eur'] = sum_spend_eur
#     return pd.Series(o_dict)

# # Define a sample DataFrame
# data = {
#     'market': ['A', 'A', 'A', 'B'],
#     'order_number': [1, 1, 2, 2],
#     'sku': ['A', 'B', 'C', 'D'],
#     'weight_kg': [10, 20, 30, 40],
#     'order_value_eur': [100, 200, 300, 400],
#     'kg_price': [5, 10, 15, 20]
# }
# df = pd.DataFrame(data)
# df
# # Call the raw_prices_agg function to calculate the raw price aggregates
# df.groupby('market sku'.split()).apply(raw_prices_agg, 'weight_kg', 'order_value_eur', 'kg_price')

# # ------------------------------------ Workflows ------------------------------------

## Benchmark and best price

# 1. Remove irrelevant markets
# 2. Lean data
#    1. Map DCs
#    2. Drop blank DCs
#    3. Get median_unit_price_loc_country_dc_sku_currency
#    4. Remove outliers based on the above median
#    5. Keep only G and ML
# 3. Get order_weight_kg
# 4. Get kg_price
# 5. Get benchmark_price_qtly_cln in a dataframe
#    1. Group by quarter, clean_name, uom
#    2. Get the wavg, values = kg_price and weight=order_weight_kg
# 6. Get best_price_qtly_cln in a dataframe
#    1. Group by quarter, (country,) dc, clean_name, uom
#    2. Get the wavg, values = kg_price and weight=order_weight_kg
#    3. Group by quarter, clean_name, uom
#    4. Get the min wavg
# 7. Create benchmark dataframe by merge on the group quarter, clean_name, uom, df + benchmark_price_qtly_cln + best_price_qtly_cln
# 8. Create country-level aggregate dataframe for data entry
#    1. Group by quarter, country, (currency,) dc, cat, subcat, fam, clean_name, uom
#    2. Get the cume_weight, wavg_price, and cume_spend in the group as a Series and cast it to a dataframe
# 9. Left-merge country-level aggreage dataframe with benchmark dataframe on quarter, clean_name, uom
# 10. Calculate savings
#     1. Current_spend - weight * benchmark_price
#     2. Current_spend - weight * best_price

# # ------------------------------------ ChatGPT ------------------------------------
# In case chatPGT does not finish code suggestion see https://www.reddit.com/r/OpenAI/comments/zgkulg/chatgpt_often_will_not_finish_its_code_or/
# Finish your answer
# Continue from the last line
# Print the rest of the code without reprinting what you've just showed me
# Finish the code in a code block. Do not print the full code again, just a missing part from last answer

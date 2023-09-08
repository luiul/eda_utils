# TODO: Write outlier removal function (based on IQR, z-score, etc.)
# TODO: Implement new mkpro function (allow user to create the directories if they don't exist).

# python libs
import os
import glob
from collections import OrderedDict
import re
from typing import List, Tuple, Union, Optional, Dict
import warnings  # to ignore (some) warnings
from pathlib import Path
import datetime as dt
import math

# data manipulation libs
import numpy as np
import pandas as pd

# data viz libs
# https://pandas.pydata.org/pandas-docs/version/1.0/user_guide/style.html
from IPython.display import display, Markdown
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

# warnings.filterwarnings("ignore")

def table(
        df: pd.DataFrame, 
        columns: Union[str, List[str]] = None, 
        n_cols:int = 3, 
        descriptive: bool = False, 
        transpose_des: bool = True, 
        corr: bool = False, 
        sns_corr: bool = False, 
        max_list_len: int = 10, 
        max_concat_list_len: int = 70,
        seed: int = 42
        ) -> None:
    """
    Prints basic dataframe stats in a tabular form, visualizes columns, and provides descriptive statistics.
    This function is designed for exploratory data analysis (EDA) to get a first overview and sample of the dataframe.

    Args:
        df (pd.DataFrame): Dataframe of interest.
        columns (Union[str, List[str]], optional): List of columns to visualize. If None, no visualization is performed.
            If 'all', visualize all columns. If a single string is passed, visualize that single column. Defaults to None.
        n_cols (int, optional): Number of columns in the grid for visualizing the columns. If set to 0, each column will be displayed in a separate plot. Defaults to 3.
        descriptive (bool, optional): If True, print descriptive statistics. Defaults to False.
        transpose_des (bool, optional): If True, transpose the descriptive statistics table. Defaults to True.
        corr (bool, optional): If True, print the correlation matrix. Defaults to False.
        sns_corr (bool, optional): If True, display a correlation matrix heatmap using Seaborn. If False, display the correlation matrix as a table. Defaults to False.
        max_list_len (int, optional): Maximum length of a list to be displayed in the "unique values" column. If the number of unique values in a column exceeds this threshold, only the count of unique values is shown. Defaults to 10.
        max_concat_list_len (int, optional): Maximum length of a concatenated list to be displayed in the "unique values" column. If the concatenated unique values string exceeds this threshold, it will be truncated and ellipses will be added. Defaults to 70.
        seed (int, optional): Seed value for reproducible sampling. Defaults to 42.

    Returns:
        None

    Displays:
        - A table containing basic statistics of the dataframe, including the number of records, column names, data types,
          the number of unique values or the count of unique values if it exceeds the threshold, the number of missing values,
          and the count of zeros or falses in each column.
        - A sample of the dataframe, with reproducible random sampling based on the seed value.
        - Descriptive statistics such as count, mean, standard deviation, minimum, quartiles, and maximum values for each numeric column in the dataframe (if `descriptive` is True).
        - A correlation matrix or a correlation matrix heatmap using Seaborn (if `corr` or `sns_corr` is True).
        - Histograms for numeric columns and bar plots for categorical columns (if `columns` is not None).

    Note:
        - The function utilizes the `tabulate` library for creating the table, and requires the `display` and `Markdown`
          modules from the IPython library for displaying the table and sample data in a Jupyter Notebook.
        - If `n_cols` is greater than 10, a warning will be issued and no plots will be created.
        - Warnings may also be issued if specified columns do not exist in the dataframe, or if all numeric or categorical
          columns have only one unique value.
    """


    rows: List[List] = []  # initialize an empty list to store rows

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
                unique_values_concat = f"{unique_values_concat[:max_concat_list_len-3]}.."  # add three dots if the concatenated values exceed the threshold
            # concatenate nunique to unique_values_concat
            unique_values_concat = f'{df[col].nunique()}/{unique_values_concat}'
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
        headers=["n", "col_name", "dtype", "nunique/u_vals", "NAs", "0s/Fs"],
        tablefmt="pipe")

     # Print the table and a sample of the dataframe
    display(Markdown(f"**Dataframe info:** Number of records: {len(df):_}"))
    # display(Markdown(table))
    print(table)
    sample = df.sample(10, random_state=seed) if len(df) > 10 else df
    display(Markdown("**Sample data:**"))
    display(sample)

    
    '''
    ===============================================================
    Display descriptive statistics if descriptive is True (default)
    ===============================================================
    '''
    if descriptive:
        # Print descriptive statistics
        display(Markdown("**Descriptive statistics:**"))
        
        # Remove count from the descriptive statistics table
        df_des = df.describe(include='all').drop('count', axis=0)

        if transpose_des: display(df_des.T)
        else: display(df_des)

    # Print information about the DataFrame including the index dtype and column dtypes, non-null values and memory usage.
    # display(Markdown("**Dataframe info:**"))
    # display(df.info(verbose=True))
    '''
    ==========================================
    Display correlation matrix if corr is True
    ==========================================
    '''
    # Print correlation matrix
    if corr and not sns_corr:
        display(Markdown("**Correlation matrix:**"))
        display(df.corr())
        sns_corr = False

    # Print correlation matrix using seaborn
    if sns_corr:
        display(Markdown("**Correlation matrix:**"))
        corr = df.corr()
        from matplotlib import MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        plt.figure(figsize=(10,8))
        plt.grid(False)  # Turn off grid lines
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='magma')
        plt.show()
        warnings.filterwarnings("default", category=MatplotlibDeprecationWarning)

    '''
    ========================================
    Visualize columns if columns is not None
    ========================================
    '''
    if columns is None: return
    # If columns is 'all', plot all columns
    if columns == 'all':
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        # Make sure that the columns exist in the dataframe
        if isinstance(columns, str): columns = [columns]

        nonexistent_cols = [col for col in columns if col not in df.columns]
        if nonexistent_cols and columns != ['all']:
            warnings.warn(f"The following columns do not exist in the dataframe: {nonexistent_cols}")

        columns = [col for col in columns if col in df.columns]
        if not columns:
            warnings.warn('No columns to plot')
            return
        numeric_cols = [col for col in columns if df[col].dtype in [np.int64, np.float64]]
        categorical_cols = [col for col in columns if df[col].dtype in ['object', 'category']]

    # Filtering columns where nunique is not 1
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]
    categorical_cols = [col for col in categorical_cols if df[col].nunique() > 1]

    # Checking if the lists are empty after filtering
    if not numeric_cols:
        warnings.warn('All numeric columns have only one unique value and have been removed')
    if not categorical_cols:
        warnings.warn('All categorical columns have only one unique value and have been removed')
    
    # Histograms for each numeric column
    if n_cols > 10: 
        warnings.warn('Too many columns to plot')
        return
    
    # Create plots instead of subplots if n_cols is 0
    if n_cols == 0: 
        # Histograms for each numeric column
        if numeric_cols:
            display(Markdown("**Histograms of numeric columns:**"))
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=df, x=col, ax=ax)
                ax.set_title(f'Histogram of {col}')
                plt.show()

        # Bar plots for each categorical column
        if categorical_cols:
            display(Markdown("**Bar plots of categorical columns:**"))
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                counts = df[col].value_counts().nlargest(20)
                sns.barplot(x=counts.index, y=counts, ax=ax, palette='magma')
                ax.set_title(f'Bar plot of {col}')
                plt.xticks(rotation=45, ha='right')
                plt.show()

    # Create subplots for numeric columns
    if numeric_cols:
        display(Markdown("**Histograms of numeric columns:**"))
        n_rows = math.ceil(len(numeric_cols) / n_cols)  # Calculate number of rows needed
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axs = axs.ravel()  # Flatten the axes array
        for i in range(n_rows * n_cols):
            if i < len(numeric_cols):
                sns.histplot(data=df, x=numeric_cols[i], ax=axs[i])
                axs[i].set_title(f'Histogram of {numeric_cols[i]}', fontsize=12)
            else:
                fig.delaxes(axs[i])  # Delete the unused axes
        plt.tight_layout()  # Adjusts subplot params to give specified padding
        plt.show()

    # Create subplots for categorical columns
    if categorical_cols:
        display(Markdown("**Bar plots of categorical columns:**"))
        n_rows = math.ceil(len(categorical_cols) / n_cols)  # Calculate number of rows needed
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axs = axs.ravel()  # Flatten the axes array
        for i in range(n_rows * n_cols):
            if i < len(categorical_cols):
                counts = df[categorical_cols[i]].value_counts().nlargest(20)
                sns.barplot(x=counts.index, y=counts, ax=axs[i], palette='magma')
                axs[i].set_title(f'Bar plot of {categorical_cols[i]}', fontsize=12)
                plt.xticks(rotation=45, ha='right')
            else:
                fig.delaxes(axs[i])  # Delete the unused axes
        plt.tight_layout()  # Adjusts subplot params to give specified padding
        plt.show()

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

def wavg(df: pd.DataFrame, values: str, weights: str) -> float:
    """
    This function computes the weighted average of a given dataframe column.

    Args:
    df (pd.DataFrame): input DataFrame.
    values (str): column in df which we want to find average of.
    weights (str): column in df which represents weights.

    Returns:
    float: Weighted average of 'values' column with respect to 'weights' column.
    """

    if not set([values, weights]).issubset(df.columns):
        raise ValueError(f"Column names provided are not in the dataframe. The dataframe has these columns: {df.columns.tolist()}")
    
    valid_df = df.dropna(subset=[values, weights])
    
    if valid_df[weights].sum() == 0:
        raise ValueError("Sum of weights is zero, cannot perform division by zero.")

    return np.average(valid_df[values], weights=valid_df[weights])

def wavg_grouped(df: pd.DataFrame, values: str, weights: str, group: Union[str, list], 
                 merge: bool = False, nan_for_zero_weights: bool = False) -> pd.DataFrame:
    """
    This function computes the weighted average of a given dataframe column within specified groups.

    Args:
    df (pd.DataFrame): input DataFrame.
    values (str): column in df which we want to find average of.
    weights (str): column in df which represents weights.
    group (Union[str, list]): column name(s) to group by. Can be a string (single column) or list of strings (multiple columns).
    merge (bool): if True, merges the input DataFrame with the resulting DataFrame.
    nan_for_zero_weights (bool): if True, returns NaN for groups where the sum of weights is zero. 

    Returns:
    pd.DataFrame: DataFrame with the weighted average of 'values' column with respect to 'weights' column for each group.
    """
    # if group is a string, convert it to list
    if isinstance(group, str):
        group = [group]

    if not set([values, weights] + group).issubset(set(df.columns)):
        raise ValueError(f"Column names provided are not in the dataframe. The dataframe has these columns: {df.columns.tolist()}")
    
    valid_df = df.dropna(subset=[values, weights] + group)

    # Check if valid_df is empty
    if valid_df.empty:
        raise ValueError("All values in the input DataFrame are missing, cannot perform weighted average.")
    
    # Check if any group has sum of weights equal to zero
    zero_weight_groups = valid_df.groupby(group).filter(lambda x: x[weights].sum() == 0)
    
    if not zero_weight_groups.empty:
        if nan_for_zero_weights:
            weighted_averages = valid_df.groupby(group).apply(lambda x: np.average(x[values], weights=x[weights]) if x[weights].sum() != 0 else np.nan)
        else:
            zero_weight_group_values = zero_weight_groups[group].drop_duplicates().values.tolist()
            raise ValueError(f"The following group(s) have sum of weights equal to zero: {zero_weight_group_values}. Cannot perform division by zero.")
    else:
        weighted_averages = valid_df.groupby(group).apply(lambda x: np.average(x[values], weights=x[weights]))

    weighted_averages = weighted_averages.reset_index().rename(columns={0: 'wavg'})
    
    if merge:
        return df.merge(weighted_averages, on=group, how='left')
    else:
        return weighted_averages

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

def mkpro(project_name: str) -> tuple:
    """
    Given a project name, create the necessary directories.

    Parameters:
    project_name (str): The name of the project.

    Returns:
    tuple: The Paths to the project, notebooks, and data directories.
    """
    # Check that the project_name is not None or empty
    if not project_name:
        print("The project_name argument is missing or empty. Please provide a valid project name.")
        return

    # Define the main project directory path
    # The directory is assumed to exist
    pdir = Path.home() / 'projects' / project_name

    # Define the notebook and data directory path
    # These will be subdirectories within the main project directory
    ndir = pdir / 'notebook'
    ddir = pdir / 'data'

    # Make sure that both subdirectories exist
    # This will create the directories if they do not exist
    # An error will be thrown if the parent directory does not exist
    for directory in [ndir, ddir]:
        directory.mkdir(exist_ok=True)
        print(f"Directory {directory} checked or created.")

    return pdir, ndir, ddir

def fpath(path, new_file='', new_root='ddir', root_idx_value='data'):
    """
    This function transforms an existing path by replacing the root directory and removing everything 
    before the new root. The new path is created using a specific root directory identifier and a new root name. 
    
    The root directory identifier is by default 'data', and the new root name is by default 'ddir'.
    
    If a new file is specified, it is added to the end of the path.

    Parameters:
    path (str): The original file path that needs to be transformed.
    new_file (str): The new file to be added at the end of the path. Default is an empty string, which means no file is added.
    new_root (str): The new root directory name, default is 'ddir'.
    root_idx_value (str): The identifier (value) of the root directory in the original path, default is 'data'.

    Returns:
    str: The transformed path.
    
    Raises:
    ValueError: If the root_idx is not found in the path.
    """
    
    # Ensure the input path is a Path object
    path = Path(path)
    
    # Split the path into parts
    parts = path.parts
    
    # Find the index of root_idx in the path parts
    try:
        root_idx = parts.index(root_idx_value)
    except ValueError:
        # Raise an error if the root_idx is not found in the path
        raise ValueError(f"The input path does not contain '{root_idx}'")
    
    # Create a new path by replacing root_idx with new_root and removing everything before root_idx
    new_parts = (new_root,) + parts[root_idx+1:]
    
    # If a new file is specified, add it to the end of the path
    if new_file:
        new_parts += (new_file,)
    
    # Join the parts back into a string, using '/' as the separator
    # Add quotation marks around each part except the new root
    new_path = '/'.join([new_parts[0]] + [f"'{part}'" for part in new_parts[1:]])
    
    # Return the new path
    print(new_path)

'''
===============================================================
Retired functions
===============================================================
'''

# def pwd() -> str:
#     """Returns the current working directory.

#     Returns:
#         A string representing the current working directory.

#     """
#     return os.getcwd()

# def touch(my_file: str) -> str:
#     """Returns the file path for a file with the specified name located in the 'data' directory of the current working
#     directory.

#     Args:
#         my_file: A string representing the name of the file to create or retrieve the path to.

#     Returns:
#         A string representing the file path for a file with the specified name located in the 'data' directory of the
#         current working directory.

#     """
#     current_directory = os.getcwd()
#     data_file_path = os.path.join(current_directory, 'data', my_file)
#     return data_file_path

# def weighted_avg(group: pd.DataFrame, weight_col: str, val_col: str) -> float:
#     """Calculate weighted average of values in a group of rows.

#     Args:
#         group (pd.DataFrame): Group of rows to calculate weighted average for.
#         weight_col (str): Name of the column to use as weights in the weighted average calculation.
#         val_col (str): Name of the column to calculate the weighted average of.

#     Returns:
#         float: Weighted average of values in the group.
#     """
#     w: pd.Series = group[weight_col]
#     v: pd.Series = group[val_col]
#     total_weight: float = w.sum()
#     if total_weight == 0:
#         raise ZeroDivisionError("The sum of weights is zero.")
#     return (w * v).sum() / total_weight

# def group_weighted_avg(df: pd.DataFrame, group_col: Union[str, List[str]],
#                        val_col: str, weight_col: str,
#                        col_name: str) -> pd.DataFrame:
#     """Group a Pandas DataFrame by a column or a list of columns and calculate the weighted average of another column.

#     Args:
#         df (pd.DataFrame): DataFrame to group and calculate weighted average for.
#         group_col (Union[str, List[str]]): Name or list of the column(s) to group the DataFrame by.
#         val_col (str): Name of the column to calculate the weighted average of.
#         weight_col (str): Name of the column to use as weights in the weighted average calculation.
#         col_name (str): Name of the resulting column.

#     Returns:
#         pd.DataFrame: DataFrame containing the weighted average of values in `val_col` for each group in the DataFrame,
#             indexed by the unique values in the `group_col` column.
#     """
#     grouped: pd.DataFrame = df.groupby(group_col, as_index=False).apply(
#         weighted_avg, weight_col, val_col)
#     grouped.columns = grouped.columns.fillna(col_name)
#     return grouped

# def group_merge_weighted_avg(
#         df: pd.DataFrame,
#         group_col: Union[str, List[str]],
#         val_col: str,
#         weight_col: str,
#         col_name: str,
#         merge_col: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
#     """Group a Pandas DataFrame by a column or a list of columns, calculate the weighted average of another column, and merge the results back to the original DataFrame.

#     Args:
#         df (pd.DataFrame): DataFrame to group and calculate weighted average for.
#         group_col (Union[str, List[str]]): Name or list of the column(s) to group the DataFrame by.
#         val_col (str): Name of the column to calculate the weighted average of.
#         weight_col (str): Name of the column to use as weights in the weighted average calculation.
#         col_name (str): Name of the resulting column.
#         merge_col (Union[str, List[str]], optional): Name or list of the column(s) to merge the resulting DataFrame on. If None, defaults to group_col. Defaults to None.

#     Returns:
#         pd.DataFrame: DataFrame containing the original columns and the new column with the weighted average for each group.
#     """
#     grouped: pd.Series = group_weighted_avg(df, group_col, val_col, weight_col,
#                                             col_name)
#     merge_col = merge_col or group_col
#     merged: pd.DataFrame = pd.merge(df, grouped, how="left", on=merge_col)
#     # merged.drop(columns=group_col, inplace=True)
#     return merged

# def weighted_operation(
#     df: pd.DataFrame, 
#     weight_col: str, 
#     value_cols: List[str], 
#     operation: str = 'mean', 
#     output_names: Dict[str, str] = None
# ) -> pd.Series:
#     """
#     Performs a specified weighted operation on the provided DataFrame.

#     Parameters:
#         df (pandas.DataFrame): A DataFrame on which to perform the operation.
#         weight_col (str): The name of the column to use as the weight.
#         value_cols (List[str]): A list of column names for the value data.
#         operation (str): The operation to perform. Default is 'mean'. Other option: 'sum'.
#         output_names (Dict[str, str]): A mapping from original column names to their names in the output. Defaults to None.

#     Returns:
#         pandas.core.series.Series: A Series containing the results of the weighted operation and the total weight.
#     """
#     data = OrderedDict()

#     weights = df[weight_col]
#     total_weight = weights.sum()

#     for value_col in value_cols:
#         values = df[value_col]

#         if operation == 'mean':
#             result = (values * weights).sum() / total_weight
#         elif operation == 'sum':
#             result = (values * weights).sum()
#         else:
#             raise ValueError(f"Unknown operation: {operation}")

#         # If output names are provided, use them, otherwise keep original names
#         if output_names:
#             data[output_names.get(value_col, value_col)] = result
#         else:
#             data[value_col] = result

#     data[f'{weight_col}_total'] = total_weight

#     return pd.Series(data)

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

# # Assume we have a DataFrame
# df = pd.DataFrame({
#     'product': ['apple', 'banana', 'apple', 'banana', 'apple'],
#     'price': [1.0, 0.5, 1.2, 0.6, 1.1],
#     'quantity': [10, 5, 20, 8, 15],
#     'weight': [0.5, 0.6, 0.7, 0.65, 0.55]
# })

# # Group by product and calculate the weighted mean and sum of price
# grouped = df.groupby('product')

# output = grouped.apply(weighted_operation, 
#                        weight_col='quantity', 
#                        value_cols=['price', 'weight'], 
#                        operation='mean', 
#                        output_names={'price': 'avg_price', 'weight': 'avg_weight'})

# print(output)

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
# Act like an <expert in field you are asking about> that / based on <some specific guideline> and <ask the thing you want it to do>

# # ------------------------------------ Useful links ------------------------------------
# Pathlib tutorial: https://github.com/Sven-Bo/pathlib-quickstart-guide/blob/master/Pathlib_Tutorial.ipynb

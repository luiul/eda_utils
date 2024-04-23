# TODO: Expand read_data_files function to remove columns after src_file column
# TODO: Change how the table fund displays list(-like) objects
# TODO: Implement a form of col_types func in the table func
# TODO: SQL connector function (with .env file and example.env in the repo)
# TODO: Write outlier removal function (based on IQR, z-score, etc.)
# TODO: Implement new mkpro function (allow user to create the directories if they don't exist).
# TODO: Create sql directory in the mkpro func
# TODO: Revise the WAVG funcs
# TODO: add loging functionality to all functions

# Standard libraries
import fnmatch
import inspect

# Logging library
import logging
import math

# Built-in libraries
import os
import warnings  # For handling warnings

# Logging and wrapping standard output to a file
from contextlib import redirect_stdout
from datetime import datetime
from functools import wraps
from pathlib import Path

# Unused libraries
# import datetime as dt
# import glob
# import re
# from collections import OrderedDict
from typing import Dict, List, Union

import matplotlib.pyplot as plt

# Data manipulation libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Data visualization libraries
from IPython.display import Markdown, display  # type: ignore

# Utility libraries
from tabulate import tabulate

# Pandas settings for better display
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:_.2f}".format

# Seaborne settings
sns.set(rc={"figure.figsize": (12, 8)})

# Uncomment to ignore warnings
# warnings.filterwarnings("ignore")

# Notes:
# 1. For formatting integers in a DataFrame:
#    df.style.format(thousands=',')
# 2. For specific column formatting in a DataFrame:
#    df.head().style.format({"col1": "{:,.0f}", "col2": "{:,.0f}"})
# More formatting options: https://pbpython.com/styling-pandas.html


def compile_daily_reports(date_str=None, report_dir='report', output_dir=None):
    """
    Compiles and collates all reports from a given day into a structured Markdown document.

    Parameters:
    - date_str (str): The date in 'YYYY-MM-DD' format to compile reports for. Defaults to today's date.
    - report_dir (str): Directory path where individual report files are stored.
    - output_dir (str): Directory path where the compiled report should be saved. If None, uses report_dir.

    Returns:
    - The filename of the compiled Markdown report.
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    if output_dir is None:
        output_dir = report_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    compiled_report_filename = os.path.join(output_dir, f"{date_str}_compiled_report.md")
    with open(compiled_report_filename, 'w') as compiled_file:
        # Write a header for the compiled report
        compiled_file.write(f"# Compiled Report for {date_str}\n\n")

        # Loop through files in the report directory
        for filename in os.listdir(report_dir):
            if date_str in filename and filename.endswith('.txt'):
                # Construct a section header for this report based on the new file naming convention
                parts = filename.split('_')
                if len(parts) >= 4:  # Ensure the filename matches the expected format
                    function_name = parts[0]
                    caller_name = parts[-3]
                    source_name = parts[-2]
                    report_title = f"{function_name} (Caller: {caller_name}, Source: {source_name})"
                else:
                    report_title = os.path.splitext(filename)[0]

                compiled_file.write(f"## {report_title}\n\n")

                # Read the individual report and add its content
                with open(os.path.join(report_dir, filename), 'r') as report_file:
                    report_content = report_file.read()
                    compiled_file.write(f"```\n{report_content}\n```\n")

    return compiled_report_filename


def log_df(df, comment=None, log_dir='report') -> pd.DataFrame:
    """
    Logs a DataFrame and an optional comment to a file, including metadata similar to the log_to_file decorator.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to log.
    - comment (str, optional): An optional comment to include in the log.
    - log_dir (str): Directory path where the log file should be saved.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The first argument must be a pandas DataFrame.")

    # Prepare metadata
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    base_caller_file = os.path.splitext(os.path.basename(caller_file))[0]
    func_name = "log_df"  # Since this is not a decorator, we manually specify the 'function' name

    # Construct the file name using the format from the decorator
    file_name = f"{timestamp}_{func_name}_{base_caller_file}.txt"
    file_path = os.path.join(log_dir, file_name)

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Log DataFrame and metadata to file
    with open(file_path, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Caller File: {caller_file}\n")
        f.write(f"Function: {func_name}\n")
        if comment:
            f.write(f"Comment: {comment}\n")
        f.write("DataFrame Output:\n")
        f.write(df.to_string())  # Convert the DataFrame to a string representation for logging

    print(f"DataFrame operation logged to: {file_path}")

    return df


def log_stdout(comment=None, log_dir='report'):
    """
    A decorator factory that allows logging of a function's output to a file, with optional comments.
    Includes metadata about the call, such as the caller and source file names, function name, and execution timestamp.

    Parameters:
    - comment (str, optional): A comment to include in the log file for additional context.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Prepare the directory and file name
            os.makedirs(log_dir, exist_ok=True)

            # Metadata collection
            src_file = inspect.getfile(func)
            try:
                caller_frame = inspect.stack()[1]
                caller_file = caller_frame.filename
            except Exception as e:
                caller_file = "unknown"
                print(f"Could not determine caller's file: {e}")
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            file_name = (
                f"{timestamp}_{func.__name__}_"
                f"{os.path.splitext(os.path.basename(caller_file))[0]}_"
                f"{os.path.splitext(os.path.basename(src_file))[0]}.txt"
            )
            file_path = os.path.join(log_dir, file_name)

            # Redirect stdout to capture print statements and function output
            with open(file_path, 'w') as f, redirect_stdout(f):
                if comment:  # Include the comment if provided
                    f.write(f"Comment: {comment}\n\n")
                f.write(
                    f"Timestamp: {timestamp}\n"
                    f"Caller File: {caller_file}\n"
                    f"Source File: {src_file}\n"
                    f"Function: {func.__name__}\n\n"
                )
                f.write("# Output:\n")
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    f.write(f"# Error: {e}\n")
                    result = None

            # Optionally, also print to stdout for immediate feedback
            with open(file_path, 'r') as f:
                content = f.read()
                print(content)

            return result

        return wrapper

    return decorator


def sanitize_df(df, include_cols=None, exclude_cols=None, upper_case_cols=None, lower_case_cols=None, verbose=True):
    """
    Sanitizes all object type columns in a given Pandas DataFrame by applying various string transformations,
    reports on aspects including non-ASCII characters (excluding missing values), missing values,
    and performs case conversions using pd.NA for missing value representation.

    Parameters:
    - df (pd.DataFrame): A Pandas DataFrame with columns to be sanitized.
    - include_cols (list, optional): List of column names to be specifically included in sanitization.
    - exclude_cols (list, optional): List of column names to be excluded from sanitization.
    - upper_case_cols (list, optional): List of column names to be converted to uppercase.
    - lower_case_cols (list, optional): List of column names to be converted to lowercase.
    - verbose (bool, optional): If True, prints detailed information about the sanitization process.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with sanitized object type columns.
    """
    # Define replacements for non-ASCII characters
    replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss', 'ñ': 'n'}

    # Initialize optional parameters as empty lists if None
    include_cols = include_cols or []
    exclude_cols = exclude_cols or []
    upper_case_cols = upper_case_cols or []
    lower_case_cols = lower_case_cols or []

    if verbose:
        # Report missing values
        # print("> Missing Values Report:")
        # missing_values = df.drop(columns=exclude_cols).isnull().sum()
        # missing_values = missing_values[missing_values > 0]
        # if not missing_values.empty:
        #     print(missing_values)
        # else:
        #     print("No missing values found in included columns.")
        # print("\n")

        # Report non-ASCII characters
        print("Non-ASCII Characters Overview:")
        for column in df.select_dtypes(include=['object']).columns:
            if column in exclude_cols:
                continue  # Skip columns in exclude_cols
            # remove NA records from df[column] and check for non-ASCII characters
            non_ascii_mask = df[column].notna() & (
                df[column] != df[column].str.encode('ascii', 'ignore').str.decode('ascii')
            )

            if non_ascii_mask.any():
                non_ascii_values = df.loc[non_ascii_mask, column]
                unique_non_ascii = non_ascii_values.unique()
                print(
                    f"Column: '{column}' has {len(unique_non_ascii)} unique non-ASCII values "
                    f"among {non_ascii_mask.sum()} occurrences."
                )
                # Print each unique non-ASCII value
                for value in unique_non_ascii:
                    print(f"    Non-ASCII value: {value}")

    # Processing columns
    for column in df.columns:
        if column in exclude_cols or df[column].dtype != 'object':
            continue

        df[column] = df[column].fillna('')  # Handle NaN values by replacing them with an empty string temporarily

        if column in upper_case_cols:
            df[column] = df[column].str.upper()
            if verbose:
                print(f"Column '{column}' converted to uppercase.")
        elif column in lower_case_cols:
            df[column] = df[column].str.lower()
            if verbose:
                print(f"Column '{column}' converted to lowercase.")
        else:
            # Apply general sanitization without changing case
            df[column] = df[column].str.strip()
            for original, replacement in replacements.items():
                df[column] = df[column].str.replace(original, replacement, regex=False)

        df[column] = df[column].replace('', np.nan)  # Revert temporary empty strings back to NaN

        if verbose:
            print(f"Sanitized column: {column}\n")

    return df


def col_types(df):
    """
    Explores the data types of the elements in each column of a DataFrame.

    For each column, this function will print:
    - The unique data types present in the column.
    - The count of each data type.

    Parameters:
    - df (pd.DataFrame): The DataFrame to explore.

    Returns:
    - None: This function prints the results and doesn't return anything.
    """
    for column in df.columns:
        print(f"Column: {column}")

        # Unique data types in the column
        unique_types = list(set(map(type, df[column])))
        print("Unique Types:", [t.__name__ for t in unique_types])

        # Count of each data type
        type_counts = df[column].map(type).value_counts().rename(lambda x: x.__name__)
        print("Type Counts:\n", type_counts)

        print("-" * 50)  # just for visual separation


# TODO: Generalize, i.e. use any series, lists, dictionary keys or values, etc. as input.
def set_diff(A, B=None):
    """
    Determine the symmetric difference details between two sets or pandas Series.

    Parameters:
    - A: First set or pandas Series
    - B: (Optional) Second set or pandas Series. If not provided, returns the set representation of A.

    Returns:
    - If B is provided: Tuple of two sets:
      1. Elements present in A but not in B
      2. Elements present in B but not in A
    - If B is not provided: Set representation of A
    """
    if isinstance(A, pd.Series):
        A = set(A)
    if B is not None:
        if isinstance(B, pd.Series):
            B = set(B)
        return (A - B, B - A)
    else:
        return A


def get_default_date_format(freq: str) -> str:
    """
    Returns the default date format based on the given frequency.
    """
    format_map = {
        "D": "%Y-%m-%d",
        "MS": "%Y-%m",
        # 'MS': '%Y-%m-%d', # Uncomment this line to return daily format for monthly start dates
        "M": "%Y-%m-%d",
        "AS": "%Y",
        "QS": "%Y-%m",
        "W": "%Y-%m-%d",
        # Add other frequencies and their formats if needed
    }

    return format_map.get(freq, "%Y-%m-%d")  # Default to daily format if not found


def expand_dates_to_range(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    dates_col: str = "expanded_dates",
    date_format: str = "auto",
    freq: str = "MS",
    inclusive: bool = False,
    inplace: bool = False,
    include_boundries: str = 'both',
) -> pd.DataFrame:
    """
    Adds a column to the DataFrame that contains a list of dates in a specified format
    between the dates in the provided start and end columns based on a specified frequency.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - start_col (str): The name of the start date column. Default is 'start_date'.
    - end_col (str): The name of the end date column. Default is 'end_date'.
    - dates_col (str): The name of the output column containing the list of dates. Default is 'expanded_dates'.
    - date_format (str): The format in which dates should be represented. Default is '%Y-%m'.
    - freq (str): The frequency for generating dates. Default is 'MS' (Month start frequency).
    - inclusive (bool): Whether to include the end_date day in the list. Default is False.
    - inplace (bool): Whether to modify the original DataFrame or return a new one. Default is False.

    Returns:
    - pd.DataFrame: The DataFrame with the added dates column.

    Examples:
    1. Monthly Start Dates (default behavior):
       - `freq`: 'MS'
       - Output: ['2021-01', '2021-02', ...]

    2. Monthly End Dates:
       - `freq`: 'M'
       - Output: ['2021-01-31', '2021-02-28', ...] when using date_format='%Y-%m-%d'

    3. Daily Dates:
       - `freq`: 'D'
       - Output: ['2021-01-01', '2021-01-02', ...] when using date_format='%Y-%m-%d'

    4. Yearly Start Dates:
       - `freq`: 'AS'
       - Output: ['2021', '2022', ...] when using date_format='%Y'

    5. Quarterly Start Dates:
       - `freq`: 'QS'
       - Output: ['2021-01', '2021-04', ...]

    6. Inclusive Monthly Start Dates:
       - `freq`: 'MS', `inclusive`: True
       - Output: ['2021-01', '2021-02', '2021-03', ...] when the end_date is within March.
    """

    # Error handling for non-existent columns
    if start_col not in df.columns or end_col not in df.columns:
        raise ValueError(f"Columns {start_col} and/or {end_col} not found in the DataFrame.")

    # If date_format is set to 'auto', derive it based on frequency
    if date_format == "auto":
        date_format = get_default_date_format(freq)

    df_copy = df.copy() if not inplace else df

    # Cast start_date and end_date to datetime
    df_copy[start_col] = pd.to_datetime(df_copy[start_col])
    df_copy[end_col] = pd.to_datetime(df_copy[end_col])

    # Function to generate a list of dates in the specified format between two dates based on a specified frequency
    def get_dates(start, end):
        if inclusive:
            end = end + pd.offsets.DateOffset(days=1)
        # Written for Pandas 1.3; note that in newer versions closed has been renamed to inclusive
        dates = pd.date_range(start=start, end=end, freq=freq, inclusive=include_boundries)  # type: ignore
        # For the 'MS' frequency, dates will be returned in our custom date format
        return dates.strftime(date_format).tolist()

    # Create the new column with the dates
    df_copy[dates_col] = df_copy.apply(lambda row: get_dates(row[start_col], row[end_col]), axis=1)

    return df_copy


def read_data_files(
    directory_path: str,
    src_col_name: str = "src_file_name",
    file_types: list = [".csv", ".tsv"],
    delimiter_map: dict = {".csv": ",", ".tsv": "\t"},  # Set default mapping directly
    encoding: str = "utf-8",
    skip_files: list = [],
    usecols: list = None,  # type: ignore
    dtype: dict = {},
    on_error: str = "warn",
    custom_readers: dict = {},
    separate: bool = False,
    add_src_file_name_column: bool = True,
    delete_empty_files: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reads data files from a directory, concatenates them into a DataFrame, or returns a dict of DataFrames.

    Parameters:
        directory_path (str): Path to the directory containing the data files.
        src_col_name (str, optional): Name for the column indicating the source file. Defaults to 'src_file_name'.
        file_types (list, optional): List of file extensions to read. Defaults to ['.csv', '.tsv'].
        delimiter_map (dict, optional): Map of file extensions to their delimiters.
        Defaults to {'.csv': ',', '.tsv': '\t'}.
        encoding (str, optional): Encoding for the files. Defaults to 'utf-8'.
        skip_files (list, optional): Filenames or patterns to skip. Defaults to [].
        usecols (list, optional): Columns to read. None means all columns. Defaults to None.
        dtype (dict, optional): Column data types. Defaults to {}.
        on_error (str, optional): Error handling ('skip', 'raise', 'warn'). Defaults to 'warn'.
        custom_readers (dict, optional): Custom functions for reading files. Defaults to {}.
        separate (bool, optional): Return separate DataFrames for each file. Defaults to False.
        add_src_file_name_column (bool, optional): Add a column with the source file name. Defaults to True.

    Returns:
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]: A single DataFrame or a dictionary of DataFrames.

    The function scans the specified directory for files matching the given extensions, reads them according to the
    specified parameters, and combines them into a single DataFrame unless 'separate' is True, in which case it returns
    a dictionary where each key is a file name and its value is the corresponding DataFrame.
    If 'add_src_file_name_column' is True, each DataFrame will include a column with the source file name.
    Errors during file reading are handled as specified by 'on_error'.
    """

    data_files = [
        f
        for f in os.listdir(directory_path)
        if any(f.endswith(ft) for ft in file_types) and not any(fnmatch.fnmatch(f, pattern) for pattern in skip_files)
    ]

    if not data_files:
        logging.warning(f"No matching files found in directory: {directory_path}")
        return {} if separate else pd.DataFrame()

    dfs = {} if separate else []

    for data_file in data_files:
        try:
            filepath = os.path.join(directory_path, data_file)
            file_ext = os.path.splitext(data_file)[-1]
            file_name = os.path.splitext(data_file)[0]

            if file_ext in custom_readers:
                df = custom_readers[file_ext](filepath)
            else:
                df = pd.read_csv(
                    filepath,
                    delimiter=delimiter_map.get(file_ext, ","),
                    encoding=encoding,
                    usecols=usecols,
                    dtype=dtype,
                )

            if not df.empty:
                if separate:
                    dfs[file_name] = df  # type: ignore
                else:
                    if add_src_file_name_column:
                        df[src_col_name] = data_file
                    dfs.append(df)  # type: ignore
            else:
                logging.warning(f"Empty DataFrame loaded from {data_file}")
                if delete_empty_files:
                    os.remove(filepath)
                    logging.info(f"Deleted empty file: {data_file}")

        except Exception as e:
            logging.error(f"Error processing {data_file}: {e}")
            if on_error == "raise":
                raise
            elif on_error == "warn":
                continue  # Proceed with the next file

    if separate:
        return dfs  # type: ignore
    else:
        return pd.concat(dfs, ignore_index=True)


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

    # Helper function to convert item to string
    def item_to_string(item):
        if isinstance(item, (list, tuple)):  # Check if item is iterable
            return ",".join(map(str, item))
        return str(item)  # If not, just convert the item to string

    # Iterate over each column and convert it to a string
    for col in cols:
        df[col] = df[col].apply(item_to_string)

    return df


def table(
    df: pd.DataFrame,
    viz_cols: Union[str, List[str]] = [],
    n_cols: int = 3,
    descriptive: bool = False,
    transpose_des: bool = True,
    corr: bool = False,
    sns_corr: bool = False,
    max_list_len: int = 10,
    max_concat_list_len: int = 70,
    seed: int = 42,
    sample_size: int = 3,
) -> None:
    """
    Prints basic dataframe stats in a tabular form, visualizes columns, and provides descriptive statistics.
    This function is designed for exploratory data analysis (EDA) to get a first overview and sample of the dataframe.

    Args:
        df (pd.DataFrame): Dataframe of interest.
        columns (Union[str, List[str]], optional): List of columns to visualize. If None, no visualization is performed.
            If 'all', visualize all columns.
            If a single string is passed, visualize that single column. Defaults to None.
        n_cols (int, optional): Number of columns in the grid for visualizing the columns.
        If set to 0, each column will be displayed in a separate plot. Defaults to 3.
        descriptive (bool, optional): If True, print descriptive statistics. Defaults to False.
        transpose_des (bool, optional): If True, transpose the descriptive statistics table. Defaults to True.
        corr (bool, optional): If True, print the correlation matrix. Defaults to False.
        sns_corr (bool, optional): If True, display a correlation matrix heatmap using Seaborn.
        If False, display the correlation matrix as a table. Defaults to False.
        max_list_len (int, optional): Maximum length of a list to be displayed in the "unique values" column.
        If the number of unique values in a column exceeds this threshold, only the count of unique values is
        shown. Defaults to 10.
        max_concat_list_len (int, optional): Maximum length of a concatenated list to be displayed in the
        "unique values" column. If the concatenated unique values string exceeds this threshold, it will be truncated
        and ellipses will be added. Defaults to 70.
        seed (int, optional): Seed value for reproducible sampling. Defaults to 42.

    Returns:
        None

    Displays:
        - A table containing basic statistics of the dataframe, including the number of records, column names, data
        types,
          the number of unique values or the count of unique values if it exceeds the threshold, the number of missing
          values,
          and the count of zeros or falses in each column.
        - A sample of the dataframe, with reproducible random sampling based on the seed value.
        - Descriptive statistics such as count, mean, standard deviation, minimum, quartiles, and maximum values for
        each numeric column in the dataframe (if `descriptive` is True).
        - A correlation matrix or a correlation matrix heatmap using Seaborn (if `corr` or `sns_corr` is True).
        - Histograms for numeric columns and bar plots for categorical columns (if `columns` is not None).

    Note:
        - The function utilizes the `tabulate` library for creating the table, and requires the `display` and `Markdown`
          modules from the IPython library for displaying the table and sample data in a Jupyter Notebook.
        - If `n_cols` is greater than 10, a warning will be issued and no plots will be created.
        - Warnings may also be issued if specified columns do not exist in the dataframe, or if all numeric or
        categorical
          columns have only one unique value.
    """

    # Identify columns that contain lists or arrays
    list_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (list, np.ndarray))]

    # Convert those columns to strings
    if list_cols:
        df = list_to_string(df, list_cols)

    rows: List[List] = []  # initialize an empty list to store rows

    # Loop through each column in the dataframe and create a row for the table
    for row_no, col in enumerate(df):
        # Assign the row number, column name, and dtype
        row: List = [row_no, col, str(df[col].dtype)]

        # Depending on the data type and number of unique values in the column, extend the row with either:
        #   - the number of unique values (if the column is an array)
        #   - the number of unique values (if the number of unique values is above the threshold)
        #   - the unique values themselves (if the number of unique values is below the threshold)
        if isinstance(df[col].iloc[0], np.ndarray):
            col_transformed: pd.Series = pd.Series(
                [",".join(map(str, arr)) for arr in df[col]]
            ).sort_values()  # convert array values to a string with elements separated by commas
            row.extend([f"{col_transformed.nunique():_}"])  # add the number of unique values to the row
            row.extend(
                [
                    f"{col_transformed.isna().sum():_}",  # add the number of NAs in the column to the row
                    f"{len(df) - np.count_nonzero(col_transformed):_}",  # add the number of zeros and falses in the
                    # column to the row
                ]
            )
        elif df[col].nunique() > max_list_len:
            row.extend([f"{df[col].nunique():_}"])  # add the number of unique values to the row
            row.extend(
                [
                    f"{df[col].isna().sum():_}",  # add the number of NAs in the column to the row
                    f"{len(df) - np.count_nonzero(df[col]):_}",  # add the number of zeros and falses in the column to
                    # the row
                ]
            )
        else:
            # unique_values: List = sorted(list(df[col].unique()))  # sort the unique values
            unique_values: List = sorted(
                [str(val) for val in df[col].unique()]
            )  # cast to string before sorting (otherwise comparisson fails)
            unique_values_concat: str = ", ".join(
                map(str, unique_values)
            )  # concatenate the unique values into a string
            if len(unique_values_concat) > max_concat_list_len:
                unique_values_concat = f"{unique_values_concat[:max_concat_list_len-3]}.."  # add three dots if the
                # concatenated values exceed the threshold
            # concatenate nunique to unique_values_concat
            unique_values_concat = f"{df[col].nunique()}/{unique_values_concat}"
            row.append(unique_values_concat)  # add the list of unique values to the row
            row.extend(
                [
                    f"{df[col].isna().sum():_}",  # add the number of NAs in the column to the row
                    f"{len(df) - np.count_nonzero(df[col]):_}",  # add the number of zeros and falses in the column to
                    # the row
                ]
            )
        # Append the row to the rows list
        rows.append(row)

    # Create and print table using the tabulate library
    table: str = tabulate(
        rows,
        headers=["n", "col_name", "dtype", "nunique/u_vals", "NAs", "0s/Fs"],
        tablefmt="pipe",
    )

    # Print the table and a sample of the dataframe
    display(Markdown(f"**Dataframe info:** Number of records: {len(df):_}"))
    # display(Markdown(table))
    print(table)
    sample = df.sample(sample_size, random_state=seed) if len(df) > sample_size else df
    display(Markdown("**Sample data:**"))
    display(sample)

    """
    ===============================================================
    Display descriptive statistics if descriptive is True (default)
    ===============================================================
    """
    if descriptive:
        # Print descriptive statistics
        display(Markdown("**Descriptive statistics:**"))

        # Remove count from the descriptive statistics table
        df_des = df.describe(include="all").drop("count", axis=0)

        if transpose_des:
            display(df_des.T)
        else:
            display(df_des)

    # Print information about the DataFrame including the index dtype and column dtypes, non-null values and memory
    # usage.
    # display(Markdown("**Dataframe info:**"))
    # display(df.info(verbose=True))
    """
    ==========================================
    Display correlation matrix if corr is True
    ==========================================
    """
    # Print correlation matrix
    if corr and not sns_corr:
        display(Markdown("**Correlation matrix:**"))
        display(df.corr())
        sns_corr = False

    # Print correlation matrix using seaborn
    if sns_corr:
        display(Markdown("**Correlation matrix:**"))
        # corr = df.corr()
        from matplotlib import MatplotlibDeprecationWarning

        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        plt.figure(figsize=(10, 8))
        plt.grid(False)  # Turn off grid lines
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="magma")
        plt.show()
        warnings.filterwarnings("default", category=MatplotlibDeprecationWarning)

    """
    ========================================
    Visualize columns if columns is not None
    ========================================
    """
    if viz_cols is []:
        return
    # If columns is 'all', plot all columns
    if viz_cols == "all":
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        # Make sure that the columns exist in the dataframe
        if isinstance(viz_cols, str):
            viz_cols = [viz_cols]

        nonexistent_cols = [col for col in viz_cols if col not in df.columns]
        if nonexistent_cols and viz_cols != ["all"]:
            warnings.warn(f"The following columns do not exist in the dataframe: {nonexistent_cols}")

        viz_cols = [col for col in viz_cols if col in df.columns]
        if not viz_cols:
            # warnings.warn("No columns to plot")
            return
        numeric_cols = [col for col in viz_cols if df[col].dtype in [np.int64, np.float64]]
        categorical_cols = [col for col in viz_cols if df[col].dtype in ["object", "category"]]

    # Filtering columns where nunique is not 1
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]
    categorical_cols = [col for col in categorical_cols if df[col].nunique() > 1]

    # Checking if the lists are empty after filtering
    if not numeric_cols:
        warnings.warn("All numeric columns have only one unique value and have been removed")
    if not categorical_cols:
        warnings.warn("All categorical columns have only one unique value and have been removed")

    # Histograms for each numeric column
    if n_cols > 10:
        warnings.warn("Too many columns to plot")
        return

    # Create plots instead of subplots if n_cols is 0
    if n_cols == 0:
        # Histograms for each numeric column
        if numeric_cols:
            display(Markdown("**Histograms of numeric columns:**"))
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=df, x=col, ax=ax)
                ax.set_title(f"Histogram of {col}")
                plt.show()

        # Bar plots for each categorical column
        if categorical_cols:
            display(Markdown("**Bar plots of categorical columns:**"))
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                counts = df[col].value_counts().nlargest(20)
                sns.barplot(x=counts.index, y=counts, ax=ax, palette="magma")
                ax.set_title(f"Bar plot of {col}")
                plt.xticks(rotation=45, ha="right")
                plt.show()

    # Create subplots for numeric columns
    if numeric_cols:
        display(Markdown("**Histograms of numeric columns:**"))
        n_rows = math.ceil(len(numeric_cols) / n_cols)  # Calculate number of rows needed
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axs = axs.ravel()  # Flatten the axes array
        for i in range(n_rows * n_cols):
            if i < len(numeric_cols):
                sns.histplot(data=df, x=numeric_cols[i], ax=axs[i])
                axs[i].set_title(f"Histogram of {numeric_cols[i]}", fontsize=12)
            else:
                fig.delaxes(axs[i])  # Delete the unused axes
        plt.tight_layout()  # Adjusts subplot params to give specified padding
        plt.show()

    # Create subplots for categorical columns
    if categorical_cols:
        display(Markdown("**Bar plots of categorical columns:**"))
        n_rows = math.ceil(len(categorical_cols) / n_cols)  # Calculate number of rows needed
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axs = axs.ravel()  # Flatten the axes array
        for i in range(n_rows * n_cols):
            if i < len(categorical_cols):
                counts = df[categorical_cols[i]].value_counts().nlargest(20)
                sns.barplot(x=counts.index, y=counts, ax=axs[i], palette="magma")
                axs[i].set_title(f"Bar plot of {categorical_cols[i]}", fontsize=12)
                plt.xticks(rotation=45, ha="right")
            else:
                fig.delaxes(axs[i])  # Delete the unused axes
        plt.tight_layout()  # Adjusts subplot params to give specified padding
        plt.show()


# Create a shorter alias for the table function
tt = table


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
            df[col] = pd.Series([", ".join(map(str, arr)) for arr in df[col]])

    return df


def flatten_multiindex(df: pd.DataFrame) -> List[str]:
    """Flatten and reverse multiindex columns

    Args:
        df (pd.DataFrame): The input DataFrame with multi-index columns

    Returns:
        List[str]: A list of column names with flattened multi-index
    """
    # Combine the first and second level column names into a single string with an underscore separator
    cols: List[str] = ["_".join(col).strip("_") for col in df.columns.values]

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
        raise ValueError(
            f"Column names provided are not in the dataframe. The dataframe has these columns: {df.columns.tolist()}"
        )

    valid_df = df.dropna(subset=[values, weights])

    if valid_df[weights].sum() == 0:
        raise ValueError("Sum of weights is zero, cannot perform division by zero.")

    return float(np.average(valid_df[values], weights=valid_df[weights]))


def wavg_grouped(
    df: pd.DataFrame,
    values: str,
    weights: str,
    group: Union[str, list],
    merge: bool = False,
    nan_for_zero_weights: bool = False,
) -> pd.DataFrame:
    """
    This function computes the weighted average of a given dataframe column within specified groups.

    Args:
    df (pd.DataFrame): input DataFrame.
    values (str): column in df which we want to find average of.
    weights (str): column in df which represents weights.
    group (Union[str, list]): column name(s) to group by. Can be a string (single column) or list of strings
    (multiple columns).
    merge (bool): if True, merges the input DataFrame with the resulting DataFrame.
    nan_for_zero_weights (bool): if True, returns NaN for groups where the sum of weights is zero.

    Returns:
    pd.DataFrame: DataFrame with the weighted average of 'values' column with respect to 'weights' column for each
    group.
    """
    # if group is a string, convert it to list
    if isinstance(group, str):
        group = [group]

    if not set([values, weights] + group).issubset(set(df.columns)):
        raise ValueError(
            f"Column names provided are not in the dataframe. The dataframe has these columns: {df.columns.tolist()}"
        )

    valid_df = df.dropna(subset=[values, weights] + group)

    # Check if valid_df is empty
    if valid_df.empty:
        raise ValueError("All values in the input DataFrame are missing, cannot perform weighted average.")

    # Check if any group has sum of weights equal to zero
    zero_weight_groups = valid_df.groupby(group).filter(lambda x: x[weights].sum() == 0)

    if not zero_weight_groups.empty:
        if nan_for_zero_weights:
            weighted_averages = valid_df.groupby(group).apply(
                lambda x: np.average(x[values], weights=x[weights]) if x[weights].sum() != 0 else np.nan  # type: ignore
            )
        else:
            zero_weight_group_values = zero_weight_groups[group].drop_duplicates().values.tolist()
            raise ValueError(
                "The following group(s) have sum of weights equal to zero: "
                + f"{zero_weight_group_values}. Cannot perform division by zero."
            )
    else:
        weighted_averages = valid_df.groupby(group).apply(
            lambda x: np.average(x[values], weights=x[weights])  # type: ignore
        )

    weighted_averages = weighted_averages.reset_index().rename(columns={0: "wavg"})

    if merge:
        return df.merge(weighted_averages, on=group, how="left")
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


# TODO: Overwrite the file if it is emtpy
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
    mapping = read_data.set_index(read_data.columns[0])[read_data.columns[1]].to_dict()
    return mapping


def create_store_col_order(file_path: str) -> list:
    """Creates a column order mapping based on an Excel table, saves it to a file, and returns a list of column names.

    If the mapping file exists at the specified file path, the function loads the mapping file and returns a list of
    column names in the order specified in the file. If the mapping file does not exist, the function reads the
    clipboard
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


def mkpro(project_path: Path = None, create_project_dir: bool = False) -> tuple:  # type: ignore
    """
    Create the necessary directories for a project.

    Parameters:
    project_path (Path): The full path to the project directory. If not provided, it's assumed to be the parent of the
    'notebook' directory.
    create_project_dir (bool): If True, the project directory will be created if it does not exist. Default is False.

    Returns:
    tuple of Path: The paths to the project, notebook, and data directories.
    """
    if project_path is None:
        current_path = Path.cwd()
        while current_path != current_path.root:
            if current_path.name == "notebook":
                project_path = current_path.parent
                break
            current_path = current_path.parent

    # Check if project_path is not an empty string, None or a non-existent directory
    if not project_path or not project_path.is_dir():
        logging.error(f"Invalid project path: {project_path}")
        return ()

    # Define the notebook and data directory path
    # ndir = project_path / "notebook"
    ddir = project_path / "data"
    sdir = project_path / "sql"

    # Create directories
    for directory in ([project_path] if create_project_dir else []) + [
        # ndir,
        ddir,
        sdir,
    ]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory {directory} checked or created.")
        except Exception as e:
            logging.error(f"Error creating directory {directory}: {e}")
            return ()

    return project_path, ddir, sdir


def fpath(path, new_file="", new_root="ddir", root_idx_value="data"):
    """
    This function transforms an existing path by replacing the root directory and removing everything
    before the new root. The new path is created using a specific root directory identifier and a new root name.

    The root directory identifier is by default 'data', and the new root name is by default 'ddir'.

    If a new file is specified, it is added to the end of the path.

    Parameters:
    path (str): The original file path that needs to be transformed.
    new_file (str): The new file to be added at the end of the path. Default is an empty string, which means no file is
    added.
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
        raise ValueError(f"The input path does not contain '{root_idx_value}'")

    # Create a new path by replacing root_idx with new_root and removing everything before root_idx
    new_parts = (new_root,) + parts[root_idx + 1 :]

    # If a new file is specified, add it to the end of the path
    if new_file:
        new_parts += (new_file,)

    # Join the parts back into a string, using '/' as the separator
    # Add quotation marks around each part except the new root
    new_path = "/".join([new_parts[0]] + [f"'{part}'" for part in new_parts[1:]])

    # Return the new path
    print(new_path)

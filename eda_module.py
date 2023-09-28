# TODO: Write outlier removal function (based on IQR, z-score, etc.)
# TODO: Implement new mkpro function (allow user to create the directories if they don't exist).
# TODO: Create sql directory in the mkpro func
# TODO: Revise the WAVG funcs

# Built-in libraries
import os
import glob
import re
import math
import fnmatch
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
import datetime as dt

# Logging library
import logging

# Data manipulation libraries
import numpy as np
import pandas as pd

# Setting pandas display options
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:_.2f}".format

# Data visualization libraries
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.figsize": (12, 8)})

# Utility libraries
from tabulate import tabulate
import warnings  # For handling warnings

# Uncomment to ignore warnings
# warnings.filterwarnings("ignore")

# Notes:
# 1. For formatting integers in a DataFrame:
#    df.style.format(thousands=',')
# 2. For specific column formatting in a DataFrame:
#    df.head().style.format({"col1": "{:,.0f}", "col2": "{:,.0f}"})
# More formatting options: https://pbpython.com/styling-pandas.html


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
    closed: str = None,
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
    - inclusive (bool): Whether to include the end_date in the list. Default is False.
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
        raise ValueError(
            f"Columns {start_col} and/or {end_col} not found in the DataFrame."
        )

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
        dates = pd.date_range(start=start, end=end, freq=freq, closed=closed)
        # For the 'MS' frequency, dates will be returned in our custom date format
        return dates.strftime(date_format).tolist()

    # Create the new column with the dates
    df_copy[dates_col] = df_copy.apply(
        lambda row: get_dates(row[start_col], row[end_col]), axis=1
    )

    return df_copy


def read_data_files(
    directory_path: str,
    source_column_name: str = "src_file",
    file_types: list = [".csv", ".tsv"],
    delimiter_map: dict = None,
    encoding: str = "utf-8",
    skip_files: list = [],
    usecols: list = None,
    dtype: dict = None,
    on_error: str = "warn",
    custom_readers: dict = None,
) -> pd.DataFrame:
    """
    Read data files from a directory and append a new column with the source file name.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing data files.
    source_column_name : str, optional
        Name of the column to append with the source file name. Default is 'source_file'.
    file_types : list, optional
        List of file extensions to consider. Default is ['.csv', '.tsv'].
    delimiter_map : dict, optional
        Dictionary mapping file extensions to delimiters.
    encoding : str, optional
        File encoding. Default is 'utf-8'.
    skip_files : list, optional
        List of filenames or patterns to skip.
    usecols : list, optional
        List of columns to read.
    dtype : dict, optional
        Dictionary specifying data types for columns.
    on_error : str, optional
        Behavior on encountering an error. Can be 'skip', 'raise', or 'warn'. Default is 'warn'.
    custom_readers : dict, optional
        Dictionary mapping file extensions to custom reading functions.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all the read files or None if no valid files were found.
    """

    if delimiter_map is None:
        delimiter_map = {".csv": ",", ".tsv": "\t"}

    data_files = [
        f
        for f in os.listdir(directory_path)
        if any(f.endswith(ft) for ft in file_types)
    ]

    if not data_files:
        logging.warning(f"No matching files found in directory: {directory_path}")
        return None

    dfs = []

    for data_file in data_files:
        # Skip files if they match any pattern in skip_files
        if any(fnmatch.fnmatch(data_file, pattern) for pattern in skip_files):
            continue

        try:
            filepath = os.path.join(directory_path, data_file)
            file_ext = os.path.splitext(data_file)[-1]

            # Use custom reader if available for the file extension
            if custom_readers and file_ext in custom_readers:
                df = custom_readers[file_ext](filepath)
            else:
                df = pd.read_csv(
                    filepath,
                    delimiter=delimiter_map.get(file_ext, ","),
                    encoding=encoding,
                    usecols=usecols,
                    dtype=dtype,
                )

            df[source_column_name] = data_file
            dfs.append(df)
            logging.info(f"Successfully processed {data_file}")

        except Exception as e:
            logging.error(f"Error processing {data_file}: {e}")
            if on_error == "raise":
                raise
            elif on_error == "warn":
                logging.warning(f"Skipped {data_file} due to error: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


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
        df[col] = pd.Series([",".join(map(str, l)) for l in df[col]])

    return df


def table(
    df: pd.DataFrame,
    columns: Union[str, List[str]] = None,
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

    # Identify columns that contain lists or arrays
    list_cols = [
        col for col in df.columns if isinstance(df[col].iloc[0], (list, np.ndarray))
    ]

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
        if type(df[col].iloc[0]) == np.ndarray:
            col_transformed: pd.Series = pd.Series(
                [",".join(map(str, l)) for l in df[col]]
            ).sort_values()  # convert array values to a string with elements separated by commas
            row.extend(
                [f"{col_transformed.nunique():_}"]
            )  # add the number of unique values to the row
            row.extend(
                [
                    f"{col_transformed.isna().sum():_}",  # add the number of NAs in the column to the row
                    f"{len(df) - np.count_nonzero(col_transformed):_}",  # add the number of zeros and falses in the column to the row
                ]
            )
        elif df[col].nunique() > max_list_len:
            row.extend(
                [f"{df[col].nunique():_}"]
            )  # add the number of unique values to the row
            row.extend(
                [
                    f"{df[col].isna().sum():_}",  # add the number of NAs in the column to the row
                    f"{len(df) - np.count_nonzero(df[col]):_}",  # add the number of zeros and falses in the column to the row
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
                unique_values_concat = f"{unique_values_concat[:max_concat_list_len-3]}.."  # add three dots if the concatenated values exceed the threshold
            # concatenate nunique to unique_values_concat
            unique_values_concat = f"{df[col].nunique()}/{unique_values_concat}"
            row.append(unique_values_concat)  # add the list of unique values to the row
            row.extend(
                [
                    f"{df[col].isna().sum():_}",  # add the number of NAs in the column to the row
                    f"{len(df) - np.count_nonzero(df[col]):_}",  # add the number of zeros and falses in the column to the row
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

    # Print information about the DataFrame including the index dtype and column dtypes, non-null values and memory usage.
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
        corr = df.corr()
        from matplotlib import MatplotlibDeprecationWarning

        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        plt.figure(figsize=(10, 8))
        plt.grid(False)  # Turn off grid lines
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma")
        plt.show()
        warnings.filterwarnings("default", category=MatplotlibDeprecationWarning)

    """
    ========================================
    Visualize columns if columns is not None
    ========================================
    """
    if columns is None:
        return
    # If columns is 'all', plot all columns
    if columns == "all":
        numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    else:
        # Make sure that the columns exist in the dataframe
        if isinstance(columns, str):
            columns = [columns]

        nonexistent_cols = [col for col in columns if col not in df.columns]
        if nonexistent_cols and columns != ["all"]:
            warnings.warn(
                f"The following columns do not exist in the dataframe: {nonexistent_cols}"
            )

        columns = [col for col in columns if col in df.columns]
        if not columns:
            warnings.warn("No columns to plot")
            return
        numeric_cols = [
            col for col in columns if df[col].dtype in [np.int64, np.float64]
        ]
        categorical_cols = [
            col for col in columns if df[col].dtype in ["object", "category"]
        ]

    # Filtering columns where nunique is not 1
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]
    categorical_cols = [col for col in categorical_cols if df[col].nunique() > 1]

    # Checking if the lists are empty after filtering
    if not numeric_cols:
        warnings.warn(
            "All numeric columns have only one unique value and have been removed"
        )
    if not categorical_cols:
        warnings.warn(
            "All categorical columns have only one unique value and have been removed"
        )

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
        n_rows = math.ceil(
            len(numeric_cols) / n_cols
        )  # Calculate number of rows needed
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
        n_rows = math.ceil(
            len(categorical_cols) / n_cols
        )  # Calculate number of rows needed
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
            df[col] = pd.Series([", ".join(map(str, l)) for l in df[col]])

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

    return np.average(valid_df[values], weights=valid_df[weights])


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
        raise ValueError(
            f"Column names provided are not in the dataframe. The dataframe has these columns: {df.columns.tolist()}"
        )

    valid_df = df.dropna(subset=[values, weights] + group)

    # Check if valid_df is empty
    if valid_df.empty:
        raise ValueError(
            "All values in the input DataFrame are missing, cannot perform weighted average."
        )

    # Check if any group has sum of weights equal to zero
    zero_weight_groups = valid_df.groupby(group).filter(lambda x: x[weights].sum() == 0)

    if not zero_weight_groups.empty:
        if nan_for_zero_weights:
            weighted_averages = valid_df.groupby(group).apply(
                lambda x: np.average(x[values], weights=x[weights])
                if x[weights].sum() != 0
                else np.nan
            )
        else:
            zero_weight_group_values = (
                zero_weight_groups[group].drop_duplicates().values.tolist()
            )
            raise ValueError(
                f"The following group(s) have sum of weights equal to zero: {zero_weight_group_values}. Cannot perform division by zero."
            )
    else:
        weighted_averages = valid_df.groupby(group).apply(
            lambda x: np.average(x[values], weights=x[weights])
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
        print(
            "The project_name argument is missing or empty. Please provide a valid project name."
        )
        return

    # Define the main project directory path
    # The directory is assumed to exist
    pdir = Path.home() / "projects" / project_name

    # Define the notebook and data directory path
    # These will be subdirectories within the main project directory
    ndir = pdir / "notebook"
    ddir = pdir / "data"

    # Make sure that both subdirectories exist
    # This will create the directories if they do not exist
    # An error will be thrown if the parent directory does not exist
    for directory in [ndir, ddir]:
        directory.mkdir(exist_ok=True)
        print(f"Directory {directory} checked or created.")

    return pdir, ndir, ddir


def fpath(path, new_file="", new_root="ddir", root_idx_value="data"):
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
    new_parts = (new_root,) + parts[root_idx + 1 :]

    # If a new file is specified, add it to the end of the path
    if new_file:
        new_parts += (new_file,)

    # Join the parts back into a string, using '/' as the separator
    # Add quotation marks around each part except the new root
    new_path = "/".join([new_parts[0]] + [f"'{part}'" for part in new_parts[1:]])

    # Return the new path
    print(new_path)

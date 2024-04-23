import ast
import os

# Specify the filename
FILENAME = "eda_module.py"


def list_functions(filename):
    """
    This function takes a filename as input and returns a list of all function names in the file.

    Parameters:
    filename (str): The name of the file to parse.

    Returns:
    list: A list of function names.
    """
    # Validate the filename to prevent path injection
    if not os.path.basename(filename) == filename or not os.path.isfile(filename):
        raise ValueError("Invalid filename")

    # Open the file in read mode
    with open(filename, "r") as source_code:
        # Parse the source code into an AST
        tree = ast.parse(source_code.read())
    # Use a list comprehension to get all function names
    funcs = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return funcs


# Get the list of function names
functions = list_functions(FILENAME)
# Print each function name in the format: "from .eda_module import {function}"
for function in functions:
    print(f"from .eda_module import {function}")

<!-- omit from toc -->
# Python EDA Utils

<!-- omit from toc -->
## Description

Collection of EDA functions for exploring, understanding, and visualizing data (including cleaning, transforming, summarizing, and visualizing data). This repo is typically used as a submodule in other repos. A complete guide can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). A summary is provided below.

<!-- omit from toc -->
## Table of Contents

- [1. How to add to an existing repo](#1-how-to-add-to-an-existing-repo)
- [2. How to clone a repo that already has submodules](#2-how-to-clone-a-repo-that-already-has-submodules)
- [3. Update a submodule](#3-update-a-submodule)
- [4. Remove a submodule](#4-remove-a-submodule)
- [5. How to use the eda\_utils in new or existing code](#5-how-to-use-the-eda_utils-in-new-or-existing-code)
  - [5.1. Importing eda\_utils in Python Scripts](#51-importing-eda_utils-in-python-scripts)
  - [5.2. Importing eda\_utils in Python Scripts](#52-importing-eda_utils-in-python-scripts)
- [6. General considerations](#6-general-considerations)
- [7. Creating a Conda environment for the submodule](#7-creating-a-conda-environment-for-the-submodule)
- [8. Create requirement for Conda environment](#8-create-requirement-for-conda-environment)
- [9. Using virtual environments](#9-using-virtual-environments)
  - [9.1. Creating a virtual environment](#91-creating-a-virtual-environment)
  - [9.2. Activating a virtual environment](#92-activating-a-virtual-environment)
  - [9.3. Installing packages](#93-installing-packages)
  - [9.4. Deactivating a virtual environment](#94-deactivating-a-virtual-environment)
  - [9.5. Deleting a virtual environment](#95-deleting-a-virtual-environment)
- [10. Creating a requirements.txt file](#10-creating-a-requirementstxt-file)
  - [10.1. Saving dependencies to requirements.txt](#101-saving-dependencies-to-requirementstxt)
  - [10.2. Installing dependencies from requirements.txt](#102-installing-dependencies-from-requirementstxt)
- [11. Misc](#11-misc)
  - [11.1. Ignore and untrack files or directories](#111-ignore-and-untrack-files-or-directories)
  - [11.2. Handling .DS\_Store files](#112-handling-ds_store-files)
- [12. References](#12-references)

## 1. How to add to an existing repo

1. Add the submodule to an existing repo:

    ```shell
    cd <parent_repo_path>
    git submodule add https://github.com/luiul/eda_utils.git eda_utils
    ```

2. Make sure that the submodule is tracking the `main`:

    ```shell
    cd eda_utils
    git checkout main
    ```

3. Add a few entries to the `.gitmodules` file. These simplify the fetching of updates from the repo tracked as submodule in the current repo. Your `.gitmodules` file should look like this:

    ```shell
    [submodule "eda_utils"]
    path = eda_utils
    url = https://github.com/luiul/eda_utils
    ignore = all
    update = merge
    branch = main
    ```

4. Commit changes to the parent repo, push etc. This will update the repo with the new submodule information

## 2. How to clone a repo that already has submodules

1. Clone the parent repo

    ```shell
    git clone <parent_repo_url>
    ```

2. At this stage, you will notice that the submodule appears as a folder in the cloned repo but it doesn't have any code. You need to update it from its remote:

    ```shell
    cd <parent_repo_path>
    git submodule init
    git submodule update
    ```

3. Make sure that the submodule is tracking the `main` (or `main`) branch

    ```shell
    cd eda_utils
    git checkout main

## 3. Update a submodule

To update the contents of a submodule, you should follow these steps:

1. Change to the submodule directory:

    ```shell
    cd <parent_repo_path>
    ```

2. Checkout the desired branch:

    ```shell
    git checkout main
    ```

3. Pull from the remote:

    ```shell
    git pull origin main
    ```

4. Change back to your project root:

    ```shell
    cd ..
    ```

5. Add the updated submodule changes:

    ```shell
    git add <parent_repo_path>
    ```

6. Commit the changes:

    ```shell
    git commit -m "Updated submodule"
    ```

7. Push the changes to your remote repository:

    ```shell
    git push origin main
    ```

## 4. Remove a submodule

1. Delete the relevant section from the `.gitmodules` file.

2. Deinitialize the submodule:

    ```shell
    git submodule deinit -f <path_to_submodule>
    ```

3. Remove the submodule from the git index and the local filesystem:

    ```shell
    git rm -f <path_to_submodule>
    ```

    If the above command results in an error, you may need to use the `--cached` option:

    ```shell
    git rm --cached <path_to_submodule>
    ```

4. Remove the actual submodule files:

    ```shell
    rm -rf .git/modules/<path_to_submodule>
    ```

5. Commit the changes:

    ```shell
    git commit -m "Removed submodule"
    ```

6. Push the changes to the remote repository:

    ```shell
    git push origin main
    ```

## 5. How to use the eda_utils in new or existing code

The submodule will appear as a subfolder structure in the parent repo. From this point all functions that exist in the `eda_utils/eda_module` folders can be imported and used in the main repo's code. For example:

```python
from eda_utils.eda_module import eda_function
```

The submodule can be utilized both in Jupyter notebooks and standalone Python scripts. If the submodule is not in the same directory as the main repo, you will need to add the submodule's parent directory to the system path before importing the submodule. See the following sections for more details.

### 5.1. Importing eda_utils in Python Scripts

To import `eda_utils` in a Jupyter notebook when the module resides in the parent directory, you can use the following code snippets:

1. With `sys.path.append()`:

    ```python
    import sys

    # Add the parent directory to the sys.path list
    sys.path.append("../")

    # Import all symbols from eda_utils module
    from eda_utils.eda_module import *
    ```

2. With `pathlib`:

    ```python
    from pathlib import Path
    import sys

    # Get the current working directory as a Path object
    current_path = Path.cwd()

    # Get the parent directory of the current working directory
    parent_path = current_path.parent

    # Convert the parent_path to a string and append it to sys.path
    sys.path.append(str(parent_path))

    # Import all symbols from eda_utils module
    from eda_utils.eda_module import *
    ```

3. With `os`:

    ```python
    import os
    import sys

    # Get the current working directory
    current_path = os.getcwd()

    # Get the parent directory of the current working directory
    parent_path = os.path.dirname(current_path)

    # Append the parent_path to sys.path
    sys.path.append(parent_path)

    # Import all symbols from eda_utils module
    from eda_utils.eda_module import *
    ```

### 5.2. Importing eda_utils in Python Scripts

If you're working within a Python script, you can import `eda_utils` as follows:

```python
import os
import sys

# get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# get the parent directory
parent_dir = os.path.dirname(script_dir)

# add the parent directory to the system path
sys.path.append(parent_dir)

# now we can import eda_utils
import eda_utils
```

This script determines the directory of the current script and its parent directory, adds the parent directory to the system path, and then imports `eda_utils`.

Please note: These solutions are quick workarounds, and they might not work in all situations. For larger and more complex projects, consider following Python packaging best practices or using a workaround with the `PYTHONPATH` environment variable.

## 6. General considerations

1. When pulling changes from remote in the **parent** repo, remember to always execute a `git submodule update --remote` command after `git pull`. `git pull` will only pull changes for the parent repo, you want to also update any changes from the submodule repo.
2. Before commiting changes from a local branch make sure you execute a `git submodule update --remote` command. This will make sure that your current commit will point to the most recent commit of the submodule.
3. To keep things simple, any changes to the `bi-isa-utils` code should be done in the original repo. You can then run a `git submodule update --remote` in any of the dependent repos to pull the changes.
4. Keep in mind that the submodule has its own `requirements.txt`. This means that, whenever you're creating a virtual environment you need to also `pip install -r eda_utils/requirements.txt`. This will install all the required packages for the submodule. If you're using Conda, you can use the following method to create a Conda environment from the submodule's `requirements.txt` file.

## 7. Creating a Conda environment for the submodule

The submodule has its own `requirements.txt` file. This means that, whenever you're creating a virtual environment, you also need to install the required packages for the submodule.

If you're using Conda, you can create an environment and install these packages using pip, as shown below:

```shell
# Create a new Conda environment
conda create -n myenv python=3.7

# Activate the environment
conda activate myenv

# Use pip to install the requirements
pip install -r eda_utils/requirements.txt
```

Replace `myenv` with your desired environment name and `3.7` with your desired Python version. `eda_utils/requirements.txt` should be replaced with the path to your `requirements.txt` file if it is located elsewhere.

If you want to use Conda for package management and you can modify the `requirements.txt` file, consider creating an `environment.yml` file. This file can specify both the Python version and the necessary packages.

Here's an example of what an `environment.yml` file might look like:

```yaml
name: myenv
channels:
  - defaults
dependencies:
  - python=3.7
  - numpy=1.18.1
  - pandas=1.0.1
  - pip:
    - -r file:requirements.txt
```

With an `environment.yml` file, you can create the environment and install all necessary packages with a single command:

```shell
conda env create -f environment.yml
```

## 8. Create requirement for Conda environment

1. Activate the desired Conda environment

    ```shell
    conda activate <env_name>
    ```

2. Export the environment's package list to a `requirements.txt` file using the `conda list` command with the `--export` flag:

    ```shell
    conda list --export > requirements.txt
    ```

Keep in mind that the `requirements.txt` file generated by Conda might not be directly compatible with `pip`. If you need a pip-compatible `requirements.txt` file, you can use the following method:

1. Install `pip` in your Conda environment if you haven't already:

    ```shell
    conda install pip
    ```

2. Use `pip freeze` to generate the `requirements.txt` file:

    ```shell
    pip freeze > requirements.txt
    ```

## 9. Using virtual environments

Working in a virtual environment is a best practice for Python development. This allows you to isolate your project and avoid conflicts between dependencies for different projects. Here's a quick guide on how you can create and use virtual environments in Python:

### 9.1. Creating a virtual environment

For Python 3, you can create a virtual environment using the `venv` module:

```shell
python3 -m venv /path/to/new/virtual/environment
```

After running this command, a directory will be created at `/path/to/new/virtual/environment` (you should replace this with the desired directory) if it doesn’t already exist. The directory will contain a Python installation; a copy of the `python` binary (or `python.exe` on Windows); and command scripts (`activate`, `deactivate`) that can be used to start and stop the environment.

### 9.2. Activating a virtual environment

You can activate the virtual environment using the `activate` script, which is located in the `bin` directory of your environment folder.

```shell
source /path/to/new/virtual/environment/bin/activate
```

When the virtual environment is activated, your shell prompt will be prefixed with the name of your environment.

### 9.3. Installing packages

Once your virtual environment is activated, you can install packages using `pip`. The packages will be installed in your virtual environment, isolated from your global Python installation.

For example, to install the requirements for your `eda_utils` submodule, you can run:

```shell
pip install -r eda_utils/requirements.txt
```

### 9.4. Deactivating a virtual environment

Once you are done working in the virtual environment, you can deactivate it:

```shell
deactivate
```

This will put you back to your system’s default Python interpreter with all its installed libraries.

To reactivate the virtual environment, just use the activation command again.

### 9.5. Deleting a virtual environment

If you want to delete a virtual environment, just delete its folder. In this case, it would be:

```shell
rm -rf /path/to/new/virtual/environment
```

Please note: this will delete all the contents in the virtual environment, including the installed packages.

## 10. Creating a requirements.txt file

A `requirements.txt` file is a file that contains a list of items that are needed to run the project. In Python, this is often a list of packages and their respective versions. Here's how you can create a `requirements.txt` file with pip:

### 10.1. Saving dependencies to requirements.txt

After setting up and activating your virtual environment, and installing all the required packages using pip (as discussed in section 10), you can save these dependencies into a `requirements.txt` file using this command:

```shell
pip freeze > requirements.txt
```

The `pip freeze` command outputs all the library packages that you installed in your project (along with their versions). The `>` operator in the shell command writes this output to a file named `requirements.txt`.

This will create a `requirements.txt` file in your project directory, listing all of the packages in the current environment, and their respective versions.

### 10.2. Installing dependencies from requirements.txt

Later, if you or someone else needs to recreate the same environment, it's as easy as using the following command:

```shell
pip install -r requirements.txt
```

This command will look at the `requirements.txt` file in your project directory and install all the dependencies listed there. This is particularly useful when you're collaborating with others or deploying your application.

Note: It's a good practice to use virtual environments when working with Python projects. This ensures that the packages required for this project won't interfere with packages for your other projects or your system Python installation.

## 11. Misc

This section provides some useful commands for handling files and directories in a Git repository.

### 11.1. Ignore and untrack files or directories

Even if a file or directory has been added to your .gitignore, Git might still track it if it was tracked previously. To untrack it, you will need to:

1. Add the file or directory to your .gitignore, if you haven't done so already, and commit this change:

    ```shell
    git add .gitignore
    git commit -m "Update .gitignore"
    ```

2. Remove the file or directory from the Git repository's tracking system, but do not delete it from your disk:

    ```shell
    git rm --cached -r [directory or file]
    ```

    Replace `[directory or file]` with the actual path to the directory or file. The `-r` option is for untracking directories recursively.

3. Commit the change:

```shell
git commit -m "Untrack files now in .gitignore"
```

### 11.2. Handling .DS_Store files

After updating your .gitignore file, you will need to remove any previously tracked `.DS_Store` files from your repository:

```shell
find . -name .DS_Store -print0 | xargs -0 git rm --cached --ignore-unmatch
```

This command finds every `.DS_Store` file in your repository and passes each one to `git rm --cached` to untrack it. The `--ignore-unmatch` option prevents `git rm` from erroring if it doesn't find a match.

Finally, commit the changes:

```shell
git commit -m "Ignore and untrack .DS_Store files"
```

## 12. References

- [Pathlib tutorial](https://github.com/Sven-Bo/pathlib-quickstart-guide/blob/master/Pathlib_Tutorial.ipynb)

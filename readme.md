<!-- omit from toc -->
# Python EDA Utils

Collection of EDA functions for exploring, understanding, and visualizing data (including cleaning, transforming, summarizing, and visualizing data). This repo is typically used as a submodule in other repos. A complete guide can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). A summary is provided below.

<!-- omit from toc -->
## Table of Contents

- [1. How to add a submodule to an existing repo](#1-how-to-add-a-submodule-to-an-existing-repo)
- [2. How to clone a repo that already has submodules](#2-how-to-clone-a-repo-that-already-has-submodules)
- [3. Update submodule to latest commit on remote](#3-update-submodule-to-latest-commit-on-remote)
- [4. Setup Git to automatically pull submodules](#4-setup-git-to-automatically-pull-submodules)
- [5. Remove submodule from parent repo (and remote)](#5-remove-submodule-from-parent-repo-and-remote)
- [6. How to use the module in your code](#6-how-to-use-the-module-in-your-code)
  - [6.1. Importing eda\_utils in Python Scripts](#61-importing-eda_utils-in-python-scripts)
  - [6.2. Importing eda\_utils in Python Scripts](#62-importing-eda_utils-in-python-scripts)
- [7. General considerations](#7-general-considerations)
- [8. Creating a Conda environment for the submodule](#8-creating-a-conda-environment-for-the-submodule)
- [9. Create requirement for Conda environment](#9-create-requirement-for-conda-environment)
- [10. Using virtual environments](#10-using-virtual-environments)
  - [10.1. Creating a virtual environment](#101-creating-a-virtual-environment)
  - [10.2. Activating a virtual environment](#102-activating-a-virtual-environment)
  - [10.3. Installing packages](#103-installing-packages)
  - [10.4. Deactivating a virtual environment](#104-deactivating-a-virtual-environment)
  - [10.5. Deleting a virtual environment](#105-deleting-a-virtual-environment)
- [11. Creating a requirements.txt file](#11-creating-a-requirementstxt-file)
  - [11.1. Saving dependencies to requirements.txt](#111-saving-dependencies-to-requirementstxt)
  - [11.2. Installing dependencies from requirements.txt](#112-installing-dependencies-from-requirementstxt)
- [12. Misc](#12-misc)
  - [12.1. Ignore and untrack files or directories](#121-ignore-and-untrack-files-or-directories)
  - [12.2. Handling .DS\_Store files](#122-handling-ds_store-files)
- [13. References](#13-references)

## 1. How to add a submodule to an existing repo

1. Add the submodule to an existing repo:

    ```shell
    cd <parent_repo_dir>
    git clone https://github.com/luiul/eda_utils.git
    git submodule add https://github.com/luiul/eda_utils.git
    ```

2. Make sure that the submodule is tracking the `main`:

    ```shell
    cd <submodule_dir>
    git checkout main
    ```

3. Add a few entries to the `.gitmodules` file. These simplify the fetching of updates from the repo tracked as submodule in the current repo. Your `.gitmodules` file should look like this:

    ```shell
    [submodule "eda_utils"]
    path = eda_utils
    url = https://github.com/luiul/eda_utils.git
    update = rebase
    branch = main
    ```

4. Commit changes to the parent repo, push etc. This will update the repo with the new submodule information

## 2. How to clone a repo that already has submodules

1. Clone the Parent Repository

   Start by cloning the parent repository. Replace `<parent_repo_url>` with the URL of the repository you wish to clone.

    ```shell
    git clone <parent_repo_url>
    ```

2. Initialize Submodules

   After cloning, submodules will appear as empty directories. You need to initialize them to prepare for updating. Navigate to the cloned repository's directory and run:

    ```shell
    cd <parent_repo_path>
    git submodule init
    ```

3. Update Submodules

   Next, fetch the content for each submodule based on the commits specified in the superproject.

    ```shell
    git submodule update
    ```

4. Simplified Initialization and Update

   Alternatively, you can initialize and update submodules in one step, including updating nested submodules recursively:

    ```shell
    git submodule update --init --recursive
    ```

5. (Optional) Update Submodules to Latest Commits

   If you wish to update all submodules to the latest commits on their respective remote branches, execute:

    ```shell
    git submodule update --recursive --remote
    ```

   This step is optional and fetches the latest changes from each submodule's remote.

6. Ensure Submodule Is Tracking the Correct Branch

   For any submodule, you might want to ensure it's tracking a specific branch (e.g., `main`). Navigate to the submodule's directory and check out the desired branch:

    ```shell
    cd <submodule_name>
    git checkout main
    ```

   Replace `<submodule_name>` with the actual name of your submodule.

**Note:** Steps 5 and 6 are optional. Step 5 updates submodules to their latest remote commits, which might not always be desired, depending on your project's requirements. Step 6 is necessary if you need the submodule to track a specific branch that differs from the one specified in the superproject.

## 3. Update submodule to latest commit on remote

To update the contents of a submodule to the latest commit on its remote repository, including the option to update recursively, follow these steps:

1. Change to the Submodule Directory:

    ```shell
    cd <submodule_dir>
    ```

2. Checkout the Desired Branch and ensure you're on the desired branch, typically `main`:

    ```shell
    git checkout main
    ```

3. Pull from the Remote:

    ```shell
    git pull origin main
    ```

4. (Optional) Recursively Update Submodules. If your submodule contains nested submodules and you wish to update all of them to their latest commits, use the following command from the submodule directory:

    ```shell
    git submodule update --recursive --remote
    ```

5. Change Back to Your Project Root, add and Commit the Updated Submodule Changes:

    ```shell
    cd ..
    git add <submodule_dir>
    git commit -m "Updated submodule to the latest commit"
    ```

6. Push the Changes:

    ```shell
    git push origin main
    ```

**Note:** The optional step provides a way to ensure that all nested submodules within your submodule are also updated to their latest commits, offering a comprehensive update across your project's dependencies. This approach simplifies managing complex projects with multiple nested submodules.

## 4. Setup Git to automatically pull submodules

When you clone a repository that contains submodules, the submodules' directories will be present, but they will initially be empty. To populate the submodules, you need to initialize them and update their contents. This can be done using the following command:

```shell
git submodule update --init --recursive
```

This command initializes your local configuration file for each submodule, updates each submodule to the commit specified by the superproject, and recursively initializes and updates each submodule within.

To automatically update all submodules when pulling in the parent repository, you can configure Git to do so with the following command:

```shell
git config --global submodule.recurse true
```

This command configures Git globally to automatically update submodules whenever you pull changes in the superproject. If you prefer to enable this behavior for a specific repository only, omit the `--global` flag and run the command within the repository:

```shell
git config submodule.recurse true
```

This setting tells Git to also pull changes for all submodules whenever you pull in the parent repository. If, however, you need to manually update the submodules to their latest commits available on their respective remote branches, use the following command:

```shell
git submodule update --recursive --remote
```

This command fetches the latest changes from the remote of each submodule and updates them to the latest commit found on their tracked branch, rather than the commit specified in the superproject.

Remember, after updating submodules, especially to newer commits not specified in the superproject, you might need to commit these changes in the superproject to track the updated submodule commits.

## 5. Remove submodule from parent repo (and remote)

1. Delete the relevant section from the `.gitmodules` file.

2. Deinitialize the submodule:

    ```shell
    git submodule deinit -f <submodule_dir>
    ```

3. Remove the submodule from the git index and the local filesystem:

    ```shell
    git rm -f <submodule_dir>
    ```

    If the above command results in an error, you may need to use the `--cached` option:

    ```shell
    git rm --cached <submodule_dir>
    ```

4. Remove the actual submodule files:

    ```shell
    rm -rf .git/modules/<submodule_dir>
    ```

5. Commit the changes:

    ```shell
    git commit -m "Removed submodule"
    ```

6. Push the changes to the remote repository:

    ```shell
    git push origin main
    ```

## 6. How to use the module in your code

The submodule will appear as a subfolder structure in the parent repo. From this point all functions that exist in the `eda_utils/eda_module` folders can be imported and used in the main repo's code. For example:

```python
from eda_utils.eda_module import eda_function
```

The submodule can be utilized both in Jupyter notebooks and standalone Python scripts. If the submodule is not in the same directory as the main repo, you will need to add the submodule's parent directory to the system path before importing the submodule. See the following sections for more details.

### 6.1. Importing eda_utils in Python Scripts

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

### 6.2. Importing eda_utils in Python Scripts

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

## 7. General considerations

1. When pulling changes from remote in the **parent** repo, remember to always execute a `git submodule update --remote` command after `git pull`. `git pull` will only pull changes for the parent repo, you want to also update any changes from the submodule repo.
2. Before commiting changes from a local branch make sure you execute a `git submodule update --remote` command. This will make sure that your current commit will point to the most recent commit of the submodule.
3. To keep things simple, any changes to the `bi-isa-utils` code should be done in the original repo. You can then run a `git submodule update --remote` in any of the dependent repos to pull the changes.
4. Keep in mind that the submodule has its own `requirements.txt`. This means that, whenever you're creating a virtual environment you need to also `pip install -r eda_utils/requirements.txt`. This will install all the required packages for the submodule. If you're using Conda, you can use the following method to create a Conda environment from the submodule's `requirements.txt` file.

## 8. Creating a Conda environment for the submodule

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

## 9. Create requirement for Conda environment

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

## 10. Using virtual environments

Working in a virtual environment is a best practice for Python development. This allows you to isolate your project and avoid conflicts between dependencies for different projects. Here's a quick guide on how you can create and use virtual environments in Python:

### 10.1. Creating a virtual environment

For Python 3, you can create a virtual environment using the `venv` module:

```shell
python3 -m venv /path/to/new/virtual/environment
```

After running this command, a directory will be created at `/path/to/new/virtual/environment` (you should replace this with the desired directory) if it doesn’t already exist. The directory will contain a Python installation; a copy of the `python` binary (or `python.exe` on Windows); and command scripts (`activate`, `deactivate`) that can be used to start and stop the environment.

### 10.2. Activating a virtual environment

You can activate the virtual environment using the `activate` script, which is located in the `bin` directory of your environment folder.

```shell
source /path/to/new/virtual/environment/bin/activate
```

When the virtual environment is activated, your shell prompt will be prefixed with the name of your environment.

### 10.3. Installing packages

Once your virtual environment is activated, you can install packages using `pip`. The packages will be installed in your virtual environment, isolated from your global Python installation.

For example, to install the requirements for your `eda_utils` submodule, you can run:

```shell
pip install -r eda_utils/requirements.txt
```

### 10.4. Deactivating a virtual environment

Once you are done working in the virtual environment, you can deactivate it:

```shell
deactivate
```

This will put you back to your system’s default Python interpreter with all its installed libraries.

To reactivate the virtual environment, just use the activation command again.

### 10.5. Deleting a virtual environment

If you want to delete a virtual environment, just delete its folder. In this case, it would be:

```shell
rm -rf /path/to/new/virtual/environment
```

Please note: this will delete all the contents in the virtual environment, including the installed packages.

## 11. Creating a requirements.txt file

A `requirements.txt` file is a file that contains a list of items that are needed to run the project. In Python, this is often a list of packages and their respective versions. Here's how you can create a `requirements.txt` file with pip:

### 11.1. Saving dependencies to requirements.txt

After setting up and activating your virtual environment, and installing all the required packages using pip (as discussed in section 10), you can save these dependencies into a `requirements.txt` file using this command:

```shell
pip freeze > requirements.txt
```

The `pip freeze` command outputs all the library packages that you installed in your project (along with their versions). The `>` operator in the shell command writes this output to a file named `requirements.txt`.

This will create a `requirements.txt` file in your project directory, listing all of the packages in the current environment, and their respective versions.

### 11.2. Installing dependencies from requirements.txt

Later, if you or someone else needs to recreate the same environment, it's as easy as using the following command:

```shell
pip install -r requirements.txt
```

This command will look at the `requirements.txt` file in your project directory and install all the dependencies listed there. This is particularly useful when you're collaborating with others or deploying your application.

Note: It's a good practice to use virtual environments when working with Python projects. This ensures that the packages required for this project won't interfere with packages for your other projects or your system Python installation.

## 12. Misc

This section provides some useful commands for handling files and directories in a Git repository.

### 12.1. Ignore and untrack files or directories

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

### 12.2. Handling .DS_Store files

After updating your .gitignore file, you will need to remove any previously tracked `.DS_Store` files from your repository:

```shell
find . -name .DS_Store -print0 | xargs -0 git rm --cached --ignore-unmatch
```

This command finds every `.DS_Store` file in your repository and passes each one to `git rm --cached` to untrack it. The `--ignore-unmatch` option prevents `git rm` from erroring if it doesn't find a match.

Finally, commit the changes:

```shell
git commit -m "Ignore and untrack .DS_Store files"
```

## 13. References

- [Pathlib tutorial](https://github.com/Sven-Bo/pathlib-quickstart-guide/blob/master/Pathlib_Tutorial.ipynb)

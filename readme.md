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
  - [5.1. Importing eda\_utils in Jupyter Notebooks](#51-importing-eda_utils-in-jupyter-notebooks)
  - [5.2. Importing eda\_utils in Python Scripts](#52-importing-eda_utils-in-python-scripts)
- [6. General considerations](#6-general-considerations)
- [7. Create requirement for Conda environment](#7-create-requirement-for-conda-environment)



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

The submodule can be utilized both in Jupyter notebooks and standalone Python scripts.

### 5.1. Importing eda_utils in Jupyter Notebooks

To import `eda_utils` in a Jupyter notebook when the module resides in the parent directory, you can use the following code snippet:

```python
import os
import sys

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

sys.path.append(parent_path)

import eda_utils
```

This script first gets the current working directory, and then gets the parent directory using `os.path.dirname()`. It then adds the parent directory to the system path before importing `eda_utils`.

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
4. Keep in mind that the submodule has its own `requirements.txt`. This means that, whenever you're creating a virtual environment you need to also `pip install -r eda_utils/requirements.txt`.

## 7. Create requirement for Conda environment

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

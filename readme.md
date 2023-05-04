# Python EDA Utils

Collection of EDA functions for exploring, understanding, and visualizing data (including cleaning, transforming, summarizing, and visualizing data). This repo is typically used as a submodule in other repos. A complete guide can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules). A summary is provided below.

## How to add to an existing repo

1. Add the submodule to an existing repo. **Rename to `eda_utils`**

    ```shell
    cd <parent_repo_path>
    git submodule add https://github.com/hellofresh/bi-isa-utils eda_utils
    ```

2. Make sure that the submodule is tracking the `master` (or `main`) branch

    ```shell
    cd eda_utils
    git checkout master
    ```

3. Add a few entries to the `.gitmodules` file. These simplify the fetching of updates from the repo tracked as submodule in the current repo. Your `.gitmodules` file should look like this:

    ```shell
    [submodule "eda_utils"]
    path = eda_utils
    url = https://github.com/luiul/eda_utils
    ignore = all
    update = merge
    branch = master
    ```

4. Commit changes to the parent repo, push etc. This will update the repo with the new submodule information

## How to clone a repo that already has submodules

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

3. Make sure that the submodule is tracking the `master` (or `main`) branch

    ```shell
    cd eda_utils
    git checkout master
    ```

## How to use the eda_utils in new or existing code

The submodule will appear as a subfolder structure in the parent repo. From this point all functions that exist in the `eda_utils/eda_module` folders can be imported and used in the main repo's code. For example:

```python
from eda_utils.eda_module import eda_function
```

## General considerations

1. When pulling changes from remote in the **parent** repo, remember to always execute a `git submodule update --remote` command after `git pull`. `git pull` will only pull changes for the parent repo, you want to also update any changes from the submodule repo.
2. Before commiting changes from a local branch make sure you execute a `git submodule update --remote` command. This will make sure that your current commit will point to the most recent commit of the submodule.
3. To keep things simple, any changes to the `bi-isa-utils` code should be done in the original repo. You can then run a `git submodule update --remote` in any of the dependent repos to pull the changes.
4. Keep in mind that the submodule has its own `requirements.txt`. This means that, whenever you're creating a virtual environment you need to also `pip install -r eda_utils/requirements.txt`.

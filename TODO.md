![repo-tree Logo](images/logo.png)
# repo-tree
## Description

`repo-tree` is a Python package for displaying a repository's tree structure. It offers a Pythonic alternative to the Linux `tree` command, with the capability to exclude files or directories based on patterns, `.gitignore`, or custom exclusion criteria.

## Features

- Display repository tree structure with customizable depth and formatting
- Exclude files and directories based on patterns or `.gitignore`
- Control visibility of hidden files
- Specify custom exclusion patterns
- Exclude files and directories whose names contain specified strings

## Installation

Install `repo-tree` directly from GitHub via pip:

```bash
pip install git+https://github.com/yotamnahum/repo-tree.git
```

Requires Python 3.6 or later.

## Usage Examples

### Basic Usage

To use `repo-tree` within a Python script:

```python
from repo_tree import RepositoryTree

# Display the current directory's tree structure
tree = RepositoryTree.display_tree(return_string=True)
print(tree)
```

Output:

```
repo-tree/
├── LICENSE
├── README.md
├── repo-tree/
│   ├── __init__.py
│   └── repository_tree.py
├── requirements.txt
└── setup.py
```

### Using `.gitignore` for Exclusions

`repo-tree` automatically excludes files and directories listed in `.gitignore`. 

Given a `.gitignore` file with:

```
*.txt
LICENSE
```

The tree output will be:

```python
tree = RepositoryTree.display_tree(return_string=True)
print(tree)
```

Output:

```
repo-tree/
├── README.md
├── repo-tree/
│   ├── __init__.py
│   └── repository_tree.py
└── setup.py
```

Note that `LICENSE` and any `.txt` files are excluded from the tree.

### Custom Exclusion Patterns and Criteria

You can provide custom exclusion patterns and criteria using the `exclusion_patterns` and `exclude_if_contains` parameters:

```python
tree = RepositoryTree.display_tree(
    max_depth=2, 
    show_hidden=False,
    exclusion_patterns=["*.pyc"],
    exclude_if_contains=["__", ".git"],
    return_string=True
)
print(tree)
```

Output:

```
repo-tree/
├── LICENSE
├── README.md
├── repo-tree/
│   └── repository_tree.py
├── requirements.txt
└── setup.py
```

In this example, the tree excludes `.pyc` files, hidden files, and files/directories containing `__` or `.git` in their names.

## Parameters

The `display_tree` method accepts the following parameters:

- `dir_path` (str): The root repository path for the tree. If not provided, the current working directory is used.
- `max_depth` (int): Maximum depth of the tree to display. Default is `float("inf")`.
- `show_hidden` (bool): Flag to show or hide hidden files. Default is `False`.
- `exclusion_patterns` (List[str], optional): Patterns to exclude from the tree.
- `exclude_if_contains` (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).
- `return_string` (bool): Flag to return the tree as a string or print it. Default is `True`.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
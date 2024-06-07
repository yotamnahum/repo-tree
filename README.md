![repo-tree Logo](images/logo.png)
# repo-tree
## Description

`repo-tree` is a Python package for displaying the tree structure of a repository. It provides a Pythonic alternative to the Linux `tree` command, with additional features such as excluding files or directories based on patterns, `.gitignore`, or custom exclusion criteria.

## Features

- Display repository tree structure with customizable depth and formatting
- Exclude files and directories based on patterns or `.gitignore`
- Control visibility of hidden files
- Specify custom exclusion patterns
- Exclude files and directories whose names contain specified strings
- Get a list of file paths in the repository tree
- Generate a concatenated view of file contents in the repository tree

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
tree = RepositoryTree.display_tree(print_tree=True)
```

Output:

```
repo-tree/
├── LICENSE
├── README.md
├── repo_tree/
│   ├── __init__.py
│   ├── cli.py
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
tree = RepositoryTree.display_tree(print_tree=True)
```

Output:

```
repo-tree/
├── README.md
├── repo_tree/
│   ├── __init__.py
│   ├── cli.py
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
    print_tree=True
)
```

Output:

```
repo-tree/
├── LICENSE
├── README.md
├── repo_tree/
│   ├── cli.py
│   └── repository_tree.py
├── requirements.txt
└── setup.py
```

In this example, the tree excludes `.pyc` files, hidden files, and files/directories containing `__` or `.git` in their names.

### Getting File Paths

You can get a list of file paths in the repository tree using the `get_tree_paths` method:

```python
file_paths = RepositoryTree.get_tree_paths(
    show_hidden=False, 
    exclusion_patterns=["*.pyc"],
    exclude_if_contains=[".git"]
)
print(file_paths)
```

Output:

```
['LICENSE', 'README.md', 'repo_tree/cli.py', 'repo_tree/repository_tree.py', 'requirements.txt', 'setup.py']
```

### Generating Concatenated File Contents

You can generate a concatenated view of file contents in the repository tree using the `get_concatenated_file_contents` method:

```python
concatenated_contents = FlatView.get_concatenated_file_contents(
    exclusion_patterns=["*TODO.md", "*.gitignore", "*setup.py"], 
    inclusion_patterns=["*.py", "*.md"]
)
print(concatenated_contents)
```

Output:

```
Below is a flattened view of the repository. Here is the repository tree structure:
    
repo-tree/
├── README.md
├── repo_tree/
│   ├── __init__.py
│   ├── cli.py
│   └── repository_tree.py

Below are the contents of all the files in the repository, separated by '###':
# README.md
[README.md contents]

###
# repo_tree/__init__.py
[__init__.py contents]

###
# repo_tree/cli.py 
[cli.py contents]

###
# repo_tree/repository_tree.py
[repository_tree.py contents]
```

This generates a flattened view of the repository with the tree structure followed by the concatenated contents of files matching the specified inclusion patterns.

## CLI Usage

`repo-tree` also provides a command-line interface (CLI) for displaying the repository tree. To use the CLI:

```bash
python -m repo_tree [dir_path] [options]
```

Options:
- `--max_depth`: Maximum depth of the tree to display (default: inf)
- `--show_hidden`: Flag to show hidden files
- `--exclusion_patterns`: Patterns to exclude from the tree

Example:

```bash
python -m repo_tree . --max_depth 2 --show_hidden --exclusion_patterns "*.pyc" "*.txt"
```

## Parameters

The `display_tree` method accepts the following parameters:

- `dir_path` (str): The root repository path for the tree. If not provided, the current working directory is used.
- `max_depth` (int): Maximum depth of the tree to display. Default is `float("inf")`.
- `show_hidden` (bool): Flag to show or hide hidden files. Default is `False`.
- `exclusion_patterns` (List[str], optional): Patterns to exclude from the tree.
- `exclude_if_contains` (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).
- `print_tree` (bool): Flag to print the tree to the console. Default is `True`.

The `get_tree_paths` and `get_concatenated_file_contents` methods accept similar parameters.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
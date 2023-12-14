![repo-tree Logo](images/logo.png)
# repo-tree
## Description

`repo-tree` is a python package for displaying a repository's tree structure. It offers a pythonic alternative to the Linux `tree` command, with the capability to exclude files or directories as specified in `.gitignore`.

## Installation

Install `repo-tree` directly from GitHub via pip:

```bash
pip install git+https://github.com/yotamnahum/repo-tree.git
```

Requires Python 3.6 or later.

## Usage Example

### Basic Usage

To use `repo-tree` within a Python script:

```python
from repo_tree import RepositoryTree

# Display the current directory's tree structure
print(RepositoryTree.display_tree())
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
print(RepositoryTree.display_tree())
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

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
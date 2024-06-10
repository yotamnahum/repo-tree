"""
Repository Tree
==============
This module provides the RepositoryTree class for representing and displaying
the tree structure of a repository, similar to the Linux 'tree' command.

Features:
- Display repository tree structure with customizable depth and formatting
- Exclude files and directories based on patterns or .gitignore
- Control visibility of hidden files
- Specify custom exclusion patterns

Author: Yotam Nahum
License: Apache License 2.0
"""

import fnmatch
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

from git import Repo

from repo_tree.utils import get_markdown_language


class RepositoryPathHandler:

    @staticmethod
    def get_absolute_path(path: Optional[str] = None) -> Path:
        """
        Get the absolute path of a given directory or the current working directory.
        Args:
            path (Optional[str]): The directory path. If None, the current working directory is used.
        Returns:
            str: The absolute path of the directory.
        """
        path = path or os.getcwd()
        if path[0].isalnum():
            path = "/" + path.lstrip("/")
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            raise Exception("Path does not exist! Enter `None` to use current working directory.")
        return Path(absolute_path)

    @staticmethod
    def is_github_url(url_or_path: str) -> bool:
        """
        A simple function to check if a given URL or path is a GitHub URL.
        Examples of valid GitHub URLs:
            - https://github.com/user/repo/tree/main/directory
            - https://github.com/user/repo
            - git@github.com:user/repo.git
        """
        return bool(re.match(r"^(https://github\.com/|git@github\.com:)", url_or_path))

    @staticmethod
    def parse_github_url(url: str) -> Tuple[str, Optional[str]]:
        match = re.match(r"^(https://github\.com/[^/]+/[^/]+)(?:/tree/[^/]+/(.+))?$", url)
        if match:
            repo_url = match.group(1)
            subdirectory = match.group(2)
            return repo_url, subdirectory
        else:
            raise ValueError(f"Invalid GitHub URL format. URL: {url}")

    @staticmethod
    def clone_github_repo(
        url: str, subdirectory: Optional[str] = None
    ) -> Tuple[tempfile.TemporaryDirectory, Optional[str]]:
        temp_dir = tempfile.TemporaryDirectory()
        repo_name = Path(url).stem
        repo_path = Path(temp_dir.name) / repo_name

        Repo.clone_from(url, repo_path)

        if subdirectory:
            subdir_path = repo_path / subdirectory
            if subdir_path.exists() and subdir_path.is_dir():
                return temp_dir, str(subdir_path)
            else:
                raise ValueError(f"Subdirectory '{subdirectory}' does not exist in the repository.")

        return temp_dir, str(repo_path)

    @classmethod
    def get_repository_path(
        cls, url_or_path: str
    ) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
        if cls.is_github_url(url_or_path):
            url, subdirectory = cls.parse_github_url(url_or_path)
            temp_dir, subdir_path = cls.clone_github_repo(url, subdirectory)
            return Path(subdir_path), temp_dir
        else:
            return cls.get_absolute_path(url_or_path), None


class TreeNode:
    _DISPLAY_PREFIX_MIDDLE = "├──"
    _DISPLAY_PREFIX_LAST = "└──"
    _PARENT_PREFIX_MIDDLE = "    "
    _PARENT_PREFIX_LAST = "│   "

    def __init__(
        self,
        path: Path,
        parent: Optional["TreeNode"] = None,
        is_last: bool = False,
    ) -> None:
        """
        Initialize a TreeNode object.

        Args:
            path (Path): The file system path for this node.
            parent (TreeNode, optional): The parent node in the repository tree.
            is_last (bool): Indicates if this node is the last sibling.
        """
        self.path: Path = Path(path)
        self.parent: Optional["TreeNode"] = parent
        self.is_last: bool = is_last
        self.depth: int = self.parent.depth + 1 if self.parent else 0

    @property
    def get_display_name(self) -> str:
        """Generate a display name for the repository tree node."""
        return f"{self.path.name}/" if self.path.is_dir() else self.path.name


class TreeGenerator:

    @staticmethod
    def string_matching_pattern(strings: Union[str, List[str]]) -> List[str]:
        if isinstance(strings, str):
            strings = [strings]
        return [f"*{string}*" for string in strings]

    @classmethod
    def _read_gitignore_patterns(cls, root: Path) -> List[str]:
        """
        Read and return the list of patterns from a .gitignore file in the given repository.

        Args:
            root (Path): Path to the directory containing the .gitignore file.

        Returns:
            List[str]: List of patterns defined in .gitignore.
        """
        ignore_patterns: List[str] = []
        gitignore_file: Path = root / ".gitignore"
        if gitignore_file.is_file():
            with gitignore_file.open("r") as file:
                ignore_patterns = [
                    line.strip() for line in file if line.strip() and not line.startswith("#")
                ]
        return ignore_patterns

    @classmethod
    def _gather_exclusion_patterns(
        cls,
        root: Path,
        exclusion_patterns: Optional[List[str]] = None,
        exclude_if_contains: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """
        Gather all exclusion patterns from various sources.

        Args:
            root (Path): The root path of the repository tree.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.
            exclude_if_contains (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).

        Returns:
            List[str]: The final list of exclusion patterns.
        """
        all_exclusion_patterns = cls._read_gitignore_patterns(root)
        all_exclusion_patterns.extend(exclusion_patterns or [])
        all_exclusion_patterns.extend(cls.string_matching_pattern(exclude_if_contains or []))
        return all_exclusion_patterns

    @staticmethod
    def _is_hidden_file(path: Path) -> bool:
        """
        Check if a file is hidden.

        Args:
            path (Path): The path of the file to check.

        Returns:
            bool: True if the file is hidden, False otherwise.
        """
        try:
            return os.stat(
                path
            ).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN or path.stem.startswith(".")
        except AttributeError:
            return path.stem.startswith(".")

    @staticmethod
    def _dirs_first(path: Path) -> Tuple[bool, str]:
        return (not path.is_dir(), str(path).lower())

    @classmethod
    def generate_tree(
        cls,
        root: Path,
        parent: Optional[TreeNode] = None,
        is_last: bool = False,
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = None,
    ) -> Generator[TreeNode, None, None]:
        """
        Build and yield nodes of the repository tree.

        Args:
            root (Path): The root path of the repository tree.
            parent (TreeNode, optional): The parent node.
            is_last (bool): Indicates if the current node is the last sibling.
            max_depth (int): The maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.

        Yields:
            Generator[TreeNode]: A generator of TreeNode objects.
        """
        root_node = TreeNode(path=root, parent=parent, is_last=is_last)
        yield root_node

        children = sorted(root.iterdir(), key=cls._dirs_first)
        if not show_hidden:
            children = [child for child in children if not cls._is_hidden_file(child)]

        children = [
            child
            for child in children
            if not any(
                fnmatch.fnmatch(str(child.relative_to(root)), pattern)
                for pattern in exclusion_patterns
            )
        ]

        for count, child in enumerate(children, start=1):
            is_last_child = count == len(children)
            if child.is_dir() and root_node.depth + 1 < max_depth:
                yield from cls.generate_tree(
                    child,
                    root_node,
                    is_last_child,
                    max_depth,
                    show_hidden,
                    exclusion_patterns,
                )
            else:
                yield TreeNode(child, root_node, is_last_child)

    @classmethod
    def build_tree(
        cls,
        dir_path: str = "",
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = None,
        exclude_if_contains: Optional[Union[str, List[str]]] = None,
    ) -> List[TreeNode]:
        """
        Build the repository tree structure.

        Args:
            dir_path (str): The root repository path for the tree.
            max_depth (int): Maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.
            exclude_if_contains (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).

        Returns:
            List[TreeNode]: The list of repository tree nodes.
        """

        all_exclusion_patterns = cls._gather_exclusion_patterns(
            dir_path, exclusion_patterns, exclude_if_contains
        )

        tree = cls.generate_tree(
            dir_path,
            max_depth=max_depth,
            show_hidden=show_hidden,
            exclusion_patterns=all_exclusion_patterns,
        )

        return [node for node in tree]


class RepositoryTree:
    @staticmethod
    def display_tree_path(node: TreeNode) -> str:
        """
        Generate the display string for the repository tree path.

        Args:
            node (TreeNode): The repository tree node.

        Returns:
            str: The formatted path string for display.
        """
        if not node.parent:
            return node.get_display_name

        prefix = TreeNode._DISPLAY_PREFIX_LAST if node.is_last else TreeNode._DISPLAY_PREFIX_MIDDLE
        parts = [f"{prefix} {node.get_display_name}"]
        parent = node.parent

        while parent and parent.parent:
            parts.append(
                TreeNode._PARENT_PREFIX_MIDDLE if parent.is_last else TreeNode._PARENT_PREFIX_LAST
            )
            parent = parent.parent

        return "".join(reversed(parts))

    @staticmethod
    def filter_included_patterns(
        tree: List[TreeNode], inclusion_patterns: List[str]
    ) -> List[TreeNode]:
        """
        Filter the nodes based on the inclusion patterns. Note that this filter affects only files, not directories.

        Args:
            tree (List[TreeNode]): The list of repository tree nodes.
            inclusion_patterns (List[str]): Patterns to include in the tree.

        Returns:
            List[TreeNode]: The filtered list of repository tree nodes.
        """
        return [
            node
            for node in tree
            if any(fnmatch.fnmatch(str(node.path), pattern) for pattern in inclusion_patterns)
            or node.path.is_dir()
        ]

    @staticmethod
    def display_tree(
        url_or_path: str = "",
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = None,
        exclude_if_contains: Optional[Union[str, List[str]]] = None,
        print_tree: bool = True,
    ) -> str:
        """
        Generate and display the directory tree.

        Args:
            url_or_path (str): The URL or path of the repository.
            max_depth (int): Maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.
            exclude_if_contains (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).
            print_tree (bool): Flag to print the tree to the console.

        Returns:
            str: The formatted tree string.

        Example:
            >>> RepositoryTree.display_tree()
            repo-tree/
            ├── images/
            │   └── logo.png
            ├── repo_tree/
            │   ├── __init__.py
            │   └── repository_tree.py
            ├── README.md
            └── setup.py
        """
        dir_path, temp_dir = RepositoryPathHandler.get_repository_path(url_or_path)
        tree = TreeGenerator.build_tree(
            dir_path, max_depth, show_hidden, exclusion_patterns, exclude_if_contains
        )
        tree_str = "\n".join(RepositoryTree.display_tree_path(node) for node in tree)
        if temp_dir:
            temp_dir.cleanup()
        if print_tree:
            print(tree_str)
        return tree_str

    @staticmethod
    def get_tree_paths(
        url_or_path: str = "",
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = None,
        exclude_if_contains: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """
        Generate and return a list of file paths in the directory tree.

        Args:
            url_or_path (str): The URL or path of the repository.
            max_depth (int): Maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.
            exclude_if_contains (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).

        Returns:
            List[str]: The list of file paths in the repository tree.

        Example:
            >>> RepositoryTree.get_tree_paths()
            ['images/logo.png', 'repo_tree/__init__.py', 'repo_tree/repository_tree.py', 'README.md', 'setup.py']
        """
        dir_path, temp_dir = RepositoryPathHandler.get_repository_path(url_or_path)

        tree = TreeGenerator.build_tree(
            dir_path, max_depth, show_hidden, exclusion_patterns, exclude_if_contains
        )

        file_paths = [str(node.path.relative_to(dir_path)) for node in tree if node.path.is_file()]
        if temp_dir:
            temp_dir.cleanup()
        return file_paths


class CodeBlock:
    def __init__(self, path: str, contents: str) -> None:
        self.path = path
        self.contents = contents


class FlatView:
    @staticmethod
    def block_formatter(code_block: CodeBlock) -> str:
        language = get_markdown_language(code_block.path)
        return f"# {code_block.path}\n```{language}\n{code_block.contents.strip()}\n```\n"

    @staticmethod
    def read_file(file_path: str, path: Path) -> CodeBlock:
        with open(path / file_path, "r", errors="ignore") as file:
            contents = file.read()
        return CodeBlock(file_path, contents)

    @staticmethod
    def format_flat_view(tree_str: str, formatted_code_blocks: List[str]) -> str:
        all_code_blocks = "\n---\n".join(formatted_code_blocks)

        template = f"""
The document below contains all the files in the codebase. For your convenience, this is the structure of the codebase:
```bash
{tree_str}
```

Below are the contents of all the files in the codebase, separated by '---'.
{all_code_blocks}
"""
        return template

    @staticmethod
    def get_concatenated_file_contents(
        url_or_path: str = "",
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = ["*TODO.md", "*.gitignore", "*setup.py"],
        exclude_if_contains: Optional[Union[str, List[str]]] = None,
        inclusion_patterns: Optional[List[str]] = ["*.py", "*.md"],
    ) -> str:
        """
        Generate and return a concatenated view of all the file contents in the directory tree.

        Args:
            url_or_path (str): The URL or path of the repository.
            max_depth (int): Maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree. Default is ['*TODO.md', '*.gitignore', '*setup.py'].
            exclude_if_contains (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).
            inclusion_patterns (List[str], optional): Patterns to include in the concatenated file contents. Default is ['*.py', '*.md'].
        Returns:
            str: The concatenated file contents.
        """
        dir_path, temp_dir = RepositoryPathHandler.get_repository_path(url_or_path)

        tree = TreeGenerator.build_tree(
            dir_path, max_depth, show_hidden, exclusion_patterns, exclude_if_contains
        )

        if inclusion_patterns:
            tree = RepositoryTree.filter_included_patterns(tree, inclusion_patterns)

        tree_str = "\n".join(RepositoryTree.display_tree_path(node) for node in tree)
        file_paths = [str(node.path.relative_to(dir_path)) for node in tree if node.path.is_file()]

        code_blocks = [FlatView.read_file(file_path, dir_path) for file_path in file_paths]
        formatted_code_blocks = [FlatView.block_formatter(block) for block in code_blocks]
        if temp_dir:
            temp_dir.cleanup()
        return FlatView.format_flat_view(tree_str, formatted_code_blocks)


def test():
    paths = [
        "https://github.com/mpoon/gpt-repository-loader/tree/main/test_data",
        "https://github.com/mpoon/gpt-repository-loader",
    ]
    for path in paths:
        print(f"Path: {path}")
        RepositoryTree.display_tree(path)


if __name__ == "__main__":
    test()

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
import stat
from pathlib import Path
from typing import Generator, List, Optional, Union


class RepositoryTree:
    """
    Class to represent and display a repository tree structure.

    Attributes:
        path (Path): The file system path for the repository tree node.
        parent (RepositoryTree, optional): The parent repository tree node.
        is_last (bool): Flag to indicate if the node is the last in its level.
        depth (int): Depth level of the node in the repository tree.
    """

    _DISPLAY_PREFIX_MIDDLE = "├──"
    _DISPLAY_PREFIX_LAST = "└──"
    _PARENT_PREFIX_MIDDLE = "    "
    _PARENT_PREFIX_LAST = "│   "

    def __init__(
        self,
        path: Path,
        parent: Optional["RepositoryTree"] = None,
        is_last: bool = False,
    ) -> None:
        """
        Initialize a RepositoryTree object.

        Args:
            path (Path): The file system path for this node.
            parent (RepositoryTree, optional): The parent node in the repository tree.
            is_last (bool): Indicates if this node is the last sibling.
        """
        self.path: Path = Path(path)
        self.parent: Optional["RepositoryTree"] = parent
        self.is_last: bool = is_last
        self.depth: int = self.parent.depth + 1 if self.parent else 0

    @property
    def display_name(self) -> str:
        """Generate a display name for the repository tree node."""
        return f"{self.path.name}/" if self.path.is_dir() else self.path.name

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

    @classmethod
    def build_tree(
        cls,
        root: Path,
        parent: Optional["RepositoryTree"] = None,
        is_last: bool = False,
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = None,
    ) -> Generator["RepositoryTree", None, None]:
        """
        Build and yield nodes of the repository tree.

        Args:
            root (Path): The root path of the repository tree.
            parent (RepositoryTree, optional): The parent node.
            is_last (bool): Indicates if the current node is the last sibling.
            max_depth (int): The maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.

        Yields:
            Generator[RepositoryTree]: A generator of RepositoryTree nodes.
        """
        root_node = cls(path=root, parent=parent, is_last=is_last)
        yield root_node

        children = sorted(root.iterdir(), key=lambda s: str(s).lower())
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
                yield from cls.build_tree(
                    child,
                    root_node,
                    is_last_child,
                    max_depth,
                    show_hidden,
                    exclusion_patterns,
                )
            else:
                yield cls(child, root_node, is_last_child)

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

    def display_path(self) -> str:
        """
        Generate the display string for the repository tree path.

        Returns:
            str: The formatted path string for display.
        """
        if not self.parent:
            return self.display_name

        prefix = self._DISPLAY_PREFIX_LAST if self.is_last else self._DISPLAY_PREFIX_MIDDLE
        parts = [f"{prefix} {self.display_name}"]
        parent = self.parent

        while parent and parent.parent:
            parts.append(self._PARENT_PREFIX_MIDDLE if parent.is_last else self._PARENT_PREFIX_LAST)
            parent = parent.parent

        return "".join(reversed(parts))

    @staticmethod
    def get_absolute_path(path: Optional[str] = None) -> str:
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
        return absolute_path

    @staticmethod
    def string_matching_pattern(strings: Union[str, List[str]]) -> List[str]:
        if isinstance(strings, str):
            strings = [strings]
        return [f"*{string}*" for string in strings]

    @staticmethod
    def display_tree(
        dir_path: str = "",
        max_depth: int = float("inf"),
        show_hidden: bool = False,
        exclusion_patterns: Optional[List[str]] = None,
        exclude_if_contains: Optional[Union[str, List[str]]] = None,
        return_string: bool = True,
    ) -> Optional[str]:
        """
        Generate and display the directory tree.

        Args:
            dir_path (str): The root repository path for the tree.
            max_depth (int): Maximum depth of the tree to display.
            show_hidden (bool): Flag to show or hide hidden files.
            exclusion_patterns (List[str], optional): Patterns to exclude from the tree.
            exclude_if_contains (Union[str, List[str]], optional): Exclude files and directories whose names contain the specified string(s).
            return_string (bool): Flag to return the tree as a string or print it.

        Returns:
            Optional[str]: The repository tree as a string if return_string is True.
        """
        path = Path(RepositoryTree.get_absolute_path(dir_path))

        all_exclusion_patterns = RepositoryTree._gather_exclusion_patterns(
            path, exclusion_patterns, exclude_if_contains
        )

        tree = RepositoryTree.build_tree(
            path,
            max_depth=max_depth,
            show_hidden=show_hidden,
            exclusion_patterns=all_exclusion_patterns,
        )

        output = "\n".join(node.display_path() for node in tree)

        if return_string:
            return output
        else:
            print(output)


def test():
    tree = RepositoryTree.display_tree(
        dir_path=".",
        max_depth=2,
        show_hidden=False,
        exclusion_patterns=["*.pyc"],
        exclude_if_contains=["__", ".git"],
        return_string=True,
    )
    print(tree)


if __name__ == "__main__":
    test()

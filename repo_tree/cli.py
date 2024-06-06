import argparse
from repository_tree import RepositoryTree

def main():
    parser = argparse.ArgumentParser(description='Display a directory tree structure.')
    parser.add_argument('dir_path', type=str, nargs='?', default='', help='The root directory path')
    parser.add_argument('--max_depth', type=int, default=float('inf'), help='Maximum depth of the tree to display')
    parser.add_argument('--show_hidden', action='store_true', help='Flag to show hidden files')
    parser.add_argument('--exclusion_patterns', type=str, nargs='*', help='Patterns to exclude from the tree')
    args = parser.parse_args()

    tree = RepositoryTree.display_tree(
        dir_path=args.dir_path,
        max_depth=args.max_depth,
        show_hidden=args.show_hidden,
        exclusion_patterns=args.exclusion_patterns,
        return_string=True
    )
    print(tree)

if __name__ == "__main__":
    main()
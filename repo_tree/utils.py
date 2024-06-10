def get_markdown_language(file_name: str) -> str:
    extension_to_language = {
        "py": "python",
        "js": "javascript",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "cs": "csharp",
        "rb": "ruby",
        "php": "php",
        "html": "html",
        "css": "css",
        "swift": "swift",
        "go": "go",
        "ts": "typescript",
        "kt": "kotlin",
        "r": "r",
        "pl": "perl",
        "sh": "shell",
        "sql": "sql",
        "scala": "scala",
        "rs": "rust",
        "yaml": "yaml",
        "json": "json",
        "xml": "xml",
        "m": "objectivec", # Also for matlab
        "dart": "dart",
        "lua": "lua",
        "hs": "haskell",
        "md": "markdown"
    }
    
    extension = file_name.split('.')[-1].lower()
    
    return extension_to_language.get(extension, "")

# Example usage:
file_name = "example.py"
print(get_markdown_language(file_name))  # Output: python

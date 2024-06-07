import unittest

from repository_tree import RepositoryPathHandler


class RepositoryPathHandlerTests(unittest.TestCase):
    def test_is_github_url(self):
        self.assertTrue(RepositoryPathHandler.is_github_url("https://github.com/user/repo"))
        self.assertTrue(RepositoryPathHandler.is_github_url("git@github.com:user/repo.git"))
        self.assertTrue(
            RepositoryPathHandler.is_github_url("https://github.com/user/repo/tree/main/directory")
        )
        self.assertFalse(RepositoryPathHandler.is_github_url("https://gitlab.com/user/repo"))

        
    def test_parse_github_url(self):
        url, subdirectory = RepositoryPathHandler.parse_github_url("https://github.com/user/repo")
        self.assertEqual(url, "https://github.com/user/repo")
        self.assertIsNone(subdirectory)

        url, subdirectory = RepositoryPathHandler.parse_github_url(
            "https://github.com/user/repo/tree/main/directory"
        )
        self.assertEqual(url, "https://github.com/user/repo")
        self.assertEqual(subdirectory, "directory")

    def test_clone_github_repo(self):
        temp_dir, subdirectory = RepositoryPathHandler.clone_github_repo(
            "https://github.com/user/repo"
        )
        self.assertIsNotNone(temp_dir)
        self.assertIsNone(subdirectory)

        temp_dir, subdirectory = RepositoryPathHandler.clone_github_repo(
            "https://github.com/user/repo/tree/main"
        )
        self.assertIsNotNone(temp_dir)
        self.assertEqual(subdirectory, "main")

    def test_get_repository_path(self):
        path, temp_dir = RepositoryPathHandler.get_repository_path("https://github.com/user/repo")
        self.assertEqual(str(path), "repo")
        self.assertIsNotNone(temp_dir)

        path, temp_dir = RepositoryPathHandler.get_repository_path(
            "https://github.com/user/repo/tree/main"
        )
        self.assertEqual(str(path), "main")
        self.assertIsNotNone(temp_dir)

        path, temp_dir = RepositoryPathHandler.get_repository_path("/path/to/repo")
        self.assertEqual(str(path), "/path/to/repo")
        self.assertIsNone(temp_dir)


if __name__ == "__main__":
    unittest.main()

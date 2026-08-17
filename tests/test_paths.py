import os
import unittest

from src.paths import configured_data_dir, configured_db_path, configured_raw_dir


class ConfiguredPathsTest(unittest.TestCase):
    def test_defaults_keep_data_under_repository(self):
        env = {}
        data_dir = configured_data_dir(env)
        self.assertTrue(data_dir.endswith(os.path.join("fantasy-projections", "data")))
        self.assertEqual(configured_db_path(env), os.path.join(data_dir, "projections.db"))
        self.assertEqual(configured_raw_dir(env), os.path.join(data_dir, "raw"))

    def test_data_dir_moves_database_and_cache_together(self):
        data_dir = os.path.abspath(os.path.join("D:\\", "fantasy-projections-data"))
        env = {"FANTASY_PROJECTIONS_DATA_DIR": data_dir}
        self.assertEqual(configured_data_dir(env), data_dir)
        self.assertEqual(configured_db_path(env), os.path.join(data_dir, "projections.db"))
        self.assertEqual(configured_raw_dir(env), os.path.join(data_dir, "raw"))

    def test_independent_overrides_take_precedence(self):
        env = {
            "FANTASY_PROJECTIONS_DATA_DIR": os.path.abspath("base"),
            "FANTASY_PROJECTIONS_DB_PATH": os.path.abspath(os.path.join("db", "custom.sqlite")),
            "FANTASY_PROJECTIONS_RAW_DIR": os.path.abspath(os.path.join("cache", "raw")),
        }
        self.assertEqual(configured_db_path(env), env["FANTASY_PROJECTIONS_DB_PATH"])
        self.assertEqual(configured_raw_dir(env), env["FANTASY_PROJECTIONS_RAW_DIR"])


if __name__ == "__main__":
    unittest.main()

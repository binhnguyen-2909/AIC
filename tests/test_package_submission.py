import tempfile
import unittest
import zipfile
from pathlib import Path

from solution.package_submission import SubmissionFormatError, package


class PackageSubmissionTest(unittest.TestCase):
    def test_packages_csv_under_submission_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "csv_results"
            input_dir.mkdir()
            (input_dir / "query-1-kis.csv").write_text("L01_V001, 1234\n", encoding="utf-8")
            (input_dir / "query-2-qa.csv").write_text(
                'L01_V001, 1234,"Có 3 người, gồm nam và nữ"\n', encoding="utf-8"
            )

            result = package(input_dir, root / "submission.zip")

            self.assertEqual([entry["rows"] for entry in result["files"]], [1, 1])
            with zipfile.ZipFile(root / "submission.zip") as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["submission/query-1-kis.csv", "submission/query-2-qa.csv"],
                )
                self.assertIn(
                    "Có 3 người, gồm nam và nữ",
                    archive.read("submission/query-2-qa.csv").decode("utf-8"),
                )

    def test_rejects_header_and_long_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "csv_results"
            input_dir.mkdir()
            kis = input_dir / "query-1-kis.csv"
            kis.write_text("video_name,frame_id\n", encoding="utf-8")
            with self.assertRaises(SubmissionFormatError):
                package(input_dir, root / "header.zip")

            kis.unlink()
            (input_dir / "query-1-qa.csv").write_text(
                f'L01_V001,1,"{"x" * 101}"\n', encoding="utf-8"
            )
            with self.assertRaises(SubmissionFormatError):
                package(input_dir, root / "long-answer.zip")


if __name__ == "__main__":
    unittest.main()

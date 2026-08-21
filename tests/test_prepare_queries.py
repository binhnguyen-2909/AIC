import json, tempfile, unittest, zipfile
from pathlib import Path
from solution.prepare_queries import prepare_queries

class PrepareQueriesTest(unittest.TestCase):
    def test_prepares_official_zip_and_extracts_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / "queries.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("query-p1-3-qa.txt", "Mô tả cảnh quay. Con số là bao nhiêu?")
                handle.writestr("query-p1-16-trake.txt", "Mở đầu.\nE1 Sự kiện một.\nE2 Sự kiện hai.\nE3 Sự kiện ba.")
                handle.writestr("query-p1-1-kis.txt", "Một cảnh quay.")
            output = root / "queries.jsonl"; result = prepare_queries(archive, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["by_type"], {"KIS": 1, "QA": 1, "TRAKE": 1})
            qa = next(row for row in rows if row["query_type"] == "QA"); self.assertEqual(qa["question"], "Con số là bao nhiêu?")
            trake = next(row for row in rows if row["query_type"] == "TRAKE"); self.assertEqual(len(trake["events"]), 3); self.assertEqual(trake["events"][1]["event"], "E2")

if __name__ == "__main__": unittest.main()

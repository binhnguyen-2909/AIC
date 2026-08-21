import csv, json, tempfile, unittest, zipfile
from pathlib import Path
from solution.make_submission import make_submission
from solution.package_submission import SubmissionFormatError

class MakeSubmissionTest(unittest.TestCase):
    def _files(self, root):
        queries = root / "queries.jsonl"
        queries.write_text("\n".join([
            json.dumps({"query_id":"K1","query_type":"KIS","query_text":"cảnh"}),
            json.dumps({"query_id":"Q1","query_type":"QA","query_text":"cảnh","question":"gì?"}),
            json.dumps({"query_id":"T1","query_type":"TRAKE","query_text":"chuỗi","events":[{"desc":"a"},{"desc":"b"}]}),
        ]) + "\n", encoding="utf-8")
        predictions = root / "predictions.jsonl"
        predictions.write_text("\n".join([
            json.dumps({"query_id":"K1","rank":1,"answer":"L01_V001, 10"}),
            json.dumps({"query_id":"Q1","rank":1,"answer":"L01_V001, 10, đỏ, xanh"}),
            json.dumps({"query_id":"T1","rank":1,"answer":"L01_V001, 10, 20"}),
        ]) + "\n", encoding="utf-8")
        return queries, predictions
    def test_converts_all_tasks_and_quotes_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); q,p=self._files(root); result=make_submission(q,p,root/"submission.zip")
            with zipfile.ZipFile(root/"submission.zip") as archive:
                self.assertEqual(len(result["members"]),3)
                qa=archive.read("submission/query-Q1-qa.csv").decode()
                self.assertEqual(next(csv.reader([qa.rstrip("\n")])),["L01_V001","10","đỏ, xanh"])
    def test_rejects_wrong_trake_event_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); q,p=self._files(root)
            p.write_text(json.dumps({"query_id":"T1","rank":1,"answer":"L01_V001, 10"})+"\n",encoding="utf-8")
            with self.assertRaises(SubmissionFormatError): make_submission(q,p,root/"submission.zip")

if __name__ == "__main__": unittest.main()

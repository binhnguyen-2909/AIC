import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from solution.repair_trake_submission import repair_submission


class RepairTrakeSubmissionTest(unittest.TestCase):
    def test_btc_context_is_counted_as_fourth_frame_only_for_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_path = root / "L24_V030.csv"
            map_path.write_text(
                "n,pts_time,fps,frame_idx\n"
                + "\n".join(f"{i + 1},{i},30,{i}" for i in range(170))
                + "\n",
                encoding="utf-8",
            )
            queries = root / "queries.jsonl"
            queries.write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "p1-1", "query_type": "KIS", "query_text": "x"}),
                        json.dumps(
                            {
                                "query_id": "p1-16",
                                "query_type": "TRAKE",
                                "query_text": "mở đầu",
                                "events": [{"event": "E1"}, {"event": "E2"}, {"event": "E3"}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "p1-1", "rank": 1, "answer": "L01_V001, 1"}),
                        json.dumps({"query_id": "p1-16", "rank": 1, "answer": "L01_V001, 1, 2, 3"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "submission.zip"

            result = repair_submission(
                queries,
                predictions,
                output,
                map_path,
                video="L24_V030",
                anchor_frames=(16, 56, 128, 143),
            )

            self.assertEqual(result["replacement_candidates"], 100)
            with zipfile.ZipFile(output) as archive:
                rows = list(
                    csv.reader(
                        archive.read("submission/query-p1-16-trake.csv")
                        .decode("utf-8")
                        .splitlines()
                    )
                )
                self.assertEqual(len(rows), 100)
                self.assertEqual(rows[0], ["L24_V030", "16", "56", "128", "143"])
                self.assertEqual(
                    list(csv.reader(archive.read("submission/query-p1-1-kis.csv").decode("utf-8").splitlines()))[0],
                    ["L01_V001", "1"],
                )

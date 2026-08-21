import unittest

from solution.preflight import validate_query_rows


class PreflightTest(unittest.TestCase):
    def test_accepts_kis_qa_and_trake_rows(self):
        rows = [
            {"query_id": "k", "query_type": "KIS", "query_text": "cảnh"},
            {"query_id": "q", "query_type": "QA", "query_text": "cảnh", "question": "màu gì?"},
            {"query_id": "t", "query_type": "TRAKE", "query_text": "chuỗi", "events": [{"desc": "một"}]},
        ]
        self.assertEqual(validate_query_rows(rows), [])

    def test_rejects_duplicate_and_incomplete_rows(self):
        rows = [
            {"query_id": "q", "query_type": "QA", "query_text": "cảnh"},
            {"query_id": "q", "query_type": "KIS", "query_text": ""},
        ]
        errors = validate_query_rows(rows)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("question" in error for error in errors))
        self.assertTrue(any("query_text" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

import unittest

from solution.submission_ens import (
    _dedupe_lines,
    _qa_answer_or_fail,
    _qa_rows_to_emit,
)


class SubmissionContractTest(unittest.TestCase):
    def test_exact_duplicate_candidates_are_removed(self):
        self.assertEqual(
            _dedupe_lines(["V, 10", "V, 10", "V, 11"], 100),
            ["V, 10", "V, 11"],
        )

    def test_blank_qa_is_fail_closed_unless_explicit_diagnostic(self):
        with self.assertRaises(RuntimeError):
            _qa_answer_or_fail("", allow_blank=False, query_id="q", rank=1)
        self.assertEqual(
            _qa_answer_or_fail("", allow_blank=True, query_id="q", rank=1), ""
        )

    def test_vlm_mode_does_not_emit_unanswered_qa_rows(self):
        results = [("V", frame, 1.0) for frame in range(100)]
        self.assertEqual(
            len(_qa_rows_to_emit(results, vlm_available=True, limit=20)), 20
        )
        self.assertEqual(
            len(_qa_rows_to_emit(results, vlm_available=False, limit=20)), 100
        )


if __name__ == "__main__":
    unittest.main()

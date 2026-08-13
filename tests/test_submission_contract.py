import unittest

from solution.submission_ens import _dedupe_lines, _qa_answer_or_fail


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


if __name__ == "__main__":
    unittest.main()

import unittest
from solution.dense_refine import _nearest_keyframe


class DenseRefineKeyframeTest(unittest.TestCase):
    def test_decoded_source_frame_is_snapped_to_nearest_keyframe(self):
        self.assertEqual(_nearest_keyframe(3173, [3054, 3190, 3320]), 3190)

    def test_empty_map_preserves_frame_for_diagnostic_fallback(self):
        self.assertEqual(_nearest_keyframe(3173, []), 3173)


if __name__ == "__main__":
    unittest.main()

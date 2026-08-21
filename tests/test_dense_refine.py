import unittest
import numpy as np
import torch
from solution.dense_refine import _device_vector

class DenseRefineTest(unittest.TestCase):
    def test_numpy_embedding_is_moved_to_torch_device(self):
        value = _device_vector(np.array([1, 2, 3], dtype=np.float64), "cpu")
        self.assertEqual(value.dtype, torch.float32); self.assertEqual(value.device.type, "cpu"); self.assertEqual(tuple(value.shape), (3,))
    def test_torch_embedding_is_flattened(self):
        value = _device_vector(torch.ones(1, 3, dtype=torch.float16), "cpu")
        self.assertEqual(value.dtype, torch.float32); self.assertEqual(tuple(value.shape), (3,))

if __name__ == "__main__": unittest.main()

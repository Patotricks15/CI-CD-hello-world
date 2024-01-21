import unittest
from sklearn.datasets import load_iris
import numpy as np


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.data = load_iris()

    def test_data_shape(self):
        self.assertEqual(self.data.data.shape, (150, 4))

    def test_no_missing_values(self):
        self.assertFalse(np.isnan(self.data.data).any())

if __name__ == '__main__':
    unittest.main()
import unittest
from machine_learning_iris.model import create_pipeline

class TestModelPerformance(unittest.TestCase):
    def test_accuracy_threshold(self):
        pipeline, X_test, y_test = create_pipeline()
        score = pipeline.score(X_test, y_test)
        self.assertGreater(score, 0.90)  # Assuming you expect at least 90% accuracy

if __name__ == '__main__':
    unittest.main()
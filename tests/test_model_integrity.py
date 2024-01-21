import unittest
from machine_learning_iris.model import create_pipeline

class TestModelPipeline(unittest.TestCase):
    def test_pipeline_data(self):
        _, _, y_test = create_pipeline()
        self.assertTrue(len(y_test) > 0)  # Verificar se o conjunto de teste não está vazio
    
    def test_pipeline_creation(self):
        pipeline, _, _ = create_pipeline()
        self.assertIsNotNone(pipeline)

    def test_model_training(self):
        _, X_test, _ = create_pipeline()
        self.assertEqual(X_test.shape[0], 45)  # 30% of 150
        
if __name__ == '__main__':
    unittest.main()
    
    

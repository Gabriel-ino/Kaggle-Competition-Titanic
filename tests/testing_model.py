from src.model import Model
import unittest
import HtmlTestRunner


class TestingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model()
    def testing_model(self):
        self.model.filter_data()
        self.model.setting_model()
    @unittest.skip("Skipping model training")
    def test_skip(self):
        self.model.fitting_model()
    
def run():
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output="./reports"))







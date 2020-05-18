from unittest import TestCase

from WKNN import WKNN

wknn = WKNN()
class TestWKNN(TestCase):
    # similarity calculation unit test
    def test___calc_similarity(self):
        self.assertTrue(wknn._WKNN__method_for_calc_similarity_unit_test())

    # distance to weight kernel unit test
    def test___inversion_kernel(self):
        self.assertTrue(wknn._WKNN__method_for_inversion_kernel_unit_test())

    # classification predict unit test
    def test___clf_predict(self):
        self.assertTrue(wknn._WKNN__method_for_clf_predict_unit_test())

    # wknn unit test
    def test___wknn(self):
        self.assertTrue(wknn._WKNN__method_for_wknn_unit_test())

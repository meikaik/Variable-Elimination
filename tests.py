import sys
import unittest
import pandas as pd

from pandas.util.testing import assert_frame_equal
from var_elimination import sumout, restrict, normalize, multiply, inference, factor_list


class TestVarElimination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f0 = {
            "X": [True, True, False, False],
            "Y": [True, False, True, False],
            "Prob": [0.1, 0.2, 0.3, 0.4],
        }
        cls.f1 = {
            "X": [True, True, True, True, False, False, False, False],
            "Y": [True, True, False, False, True, True, False, False],
            "Z": [True, False, True, False, True, False, True, False],
            "Prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }

    def testRestrict(self):
        f2 = {
            "Y": [True, True, False, False],
            "Z": [True, False, True, False],
            "Prob": [0.1, 0.2, 0.3, 0.4],
        }
        df1 = pd.DataFrame(self.f1)
        expected = pd.DataFrame(f2)
        actual = restrict(df1, "X", True)
        assert_frame_equal(expected, actual)

    def testSumout(self):
        f2 = {
            "Y": [False, False, True, True],
            "Z": [False, True, False, True],
            "Prob": [1.2, 1.0, 0.8, 0.6],
        }
        df1 = pd.DataFrame(self.f1)
        expected = pd.DataFrame(f2)
        actual = sumout(df1, "X")
        assert_frame_equal(actual, expected)

    def testSumoutSingleCol(self):
        f1 = {
            "X": [True, True, True, True, False, False, False, False],
            "Prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
        f2 = {
            "Prob": [3.6],
        }
        df1 = pd.DataFrame(f1)
        expected = pd.DataFrame(f2)
        actual = sumout(df1, "X")
        assert_frame_equal(expected, actual)

    def testNormalize(self):
        expected = pd.DataFrame(self.f0)
        actual = normalize(expected)
        assert_frame_equal(expected, actual)

    def testMultiply(self):
        f1 = {
            "X": [True, True, False, False],
            "Y": [True, False, True, False],
            "Prob": [0.1, 0.2, 0.3, 0.4],
        }
        f2 = {
            "Y": [True, False, True, False],
            "Z": [True, True, False, False],
            "Prob": [0.4, 0.3, 0.2, 0.1],
        }
        f3 = {
            "X": [True, True, False, False, True, True, False, False],
            "Y": [True, True, True, True, False, False, False, False],
            "Z": [True, False, True, False, True, False, True, False],
            "Prob": [0.04, 0.02, 0.12, 0.06, 0.06, 0.02, 0.12, 0.04],
        }
        df1 = pd.DataFrame(f1)
        df2 = pd.DataFrame(f2)
        expected = pd.DataFrame(f3)
        actual = multiply(df1, df2)
        assert_frame_equal(expected, actual)

    def testMultiplyConstant(self):
        f2 = {"Prob": [0.1]}
        f3 = {
            "X": [True, True, False, False],
            "Y": [True, False, True, False],
            "Prob": [0.01, 0.02, 0.03, 0.04],
        }
        df1 = pd.DataFrame(self.f0)
        df2 = pd.DataFrame(f2)
        expected = pd.DataFrame(f3)
        actual = multiply(df1, df2)
        assert_frame_equal(expected, actual)

    def testInferenceA(self):
        f1 = {
            "FH": [False, True],
            "Prob": [0.92693, 0.07307],
        }
        expected = pd.DataFrame(f1)
        actual = inference(factor_list(), ["FH"])
        assert_frame_equal(expected, actual)


    def testInferenceB(self):
        f1 = {
            "FS": [True, False],
            "Prob": [0.08594147120761021, 0.9140585287923899],
        }
        expected = pd.DataFrame(f1)
        actual = inference(factor_list(), ["FS"], ["FM", "FH"])
        assert_frame_equal(expected, actual)

    def testInferenceC(self):
        f1 = {
            "FS": [False, True],
            "Prob": [0.639333, 0.360667],
        }
        expected = pd.DataFrame(f1)
        actual = inference(factor_list(), ["FS"], ["FM", "FH", "FB"])
        assert_frame_equal(expected, actual)

    def testInferenceD(self):
        f1 = {
            "FH": [False, True],
            "Prob": [0.92693, 0.07307],
        }
        expected = pd.DataFrame(f1)
        actual = inference(factor_list(), ["FH"])
        assert_frame_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()

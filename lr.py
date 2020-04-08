import numpy as np 
__all__ = ['LogisticRegression']

class LogisticRegression(object):
    @staticmethod
    @np.vectorize
    def sigmoid(x):
        #sigmoid_rangeで浮動小数点エラー対策
        sigmoid_range = 34.538776394910684
        #clip(検索値,検索値<-sigmoid_rangeの時、sigmoid_range<検索値)
        x = np.clip(x, -sigmoid_range, sigmoid_range)

        #piecewise(検索値,条件,出力値)
        return 1.0 / (1.0 + np.exp(-x))
import numpy as np
from abc import ABCMeta, abstractmethod

__all__ = ['BaseBinaryNaiveBayes', 'NaiveBayes1']

class BaseBinaryNaiveBayes(object, metaclass=ABCMeta):

    def __init__(self):
        """
        Constructor
        """
        self.pY_ = None
        self.pXgY_ = None

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):

        #Xの行数(訓練事例数)を代入
        n_samples = X.shape[0]
        #Xの列数(特微数)を代入
        n_features = X.shape[1]
        #y:各特徴ベクトルに対する予測ベクトルをまとめた配列
        y = np.empty(n_samples, dtype=int)
        
        for i, xi in enumerate(X):

            logpXY = (np.log(self.pY_) + np.sum(np.log(self.pXgY_[np.arange(n_features), xi, :]),axis=0))

            y[i] = np.argmax(logpXY)
        
        return y


class NaiveBayes1(BaseBinaryNaiveBayes):
    """
    Naive Bayes class (1)
    """
    def __init__(self):
        super(NaiveBayes1, self).__init__()

    #X:行数が訓練事例数,列数が特微数からなる2次元配列
    #y:クラスラベル集合の１次元配列
    def fit(self, X, y):

        #Xの行数(訓練事例数)を代入
        n_samples = X.shape[0]
        #Xの列数(特微数)を代入
        n_features = X.shape[1]

        #クラス数
        n_classes = 2
        #特微数
        n_fvalues = 2

        #特微数とクラスラベル事例数が一致しているか確認
        if n_samples != len(y):
            raise ValueError('Mismatched number of samples.')

    #クラス分布
        #nYは各事例を数える箱
        #nYの初期宣言(大きさはクラスの数)
        ary_y = y[:, np.newaxis]
        ary_yi = np.arange(n_classes)[np.newaxis, :]

        nY = np.sum(ary_y == ary_yi, axis=0)

        #各クラスの確率を計算してpYに格納
        self.pY_ = np.empty(n_classes, dtype=float)
        self.pY_ = nY = n_samples

    #特徴分布
        ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]
        ary_X = X[:, :, np.newaxis, np.newaxis]
        ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
        ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]

        nXY = np.sum(np.logical_and(ary_y == ary_yi, ary_X == ary_xi), axis=0)

        self.pXgY_ = nXY / nXY.sum(axis=1, keepdims=True)

        pass
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


__all__ = ['NaiveBayes1']

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
        cmp_y = (ary_y == ary_yi)
        nY = np.sum(cmp_y, axis=0)

        #各クラスの確率を計算してpYに格納
        self.pY_ = np.empty(n_classes, dtype=float)
        for i in range(n_classes):
            self.pY_[i] = nY[i] / n_samples

    #特徴分布
        nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=int)
        for i in range(n_samples):
            for j in range(n_features):
                nXY[j, X[i, j], y[i]] += 1

        self.pXgY_ = np.empty((n_features, n_fvalues, n_classes),dtype=float)
        for j in range(n_features):
            for xi in range(n_fvalues):
                for yi in range(n_classes):
                    self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / float(nY[yi])
        pass
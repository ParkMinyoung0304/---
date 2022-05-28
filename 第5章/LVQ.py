# 创建LVQ类
import copy
import numpy as np
class LVQ():
    def __init__(self, max_iter=10000, eta=0.1, e=0.01):
        self.max_iter = max_iter
        self.eta = eta
        self.e = e
    # 计算欧式距离
    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)
    # 产生原型向量
    def get_vector(self, x, y):
        k = len(set(y))
        index = np.random.choice(x.shape[0], 1, replace=False)
        mus = []
        mus.append(x[index])
        mus_label = []
        mus_label.append(y[index])
        for _ in range(k - 1):
            max_dist_index = 0
            max_distance = 0
            for j in range(x.shape[0]):
                min_dist_with_mu = 999999
                for mu in mus:
                    dist_with_mu = self.dist(mu, x[j])
                    if min_dist_with_mu > dist_with_mu:
                        min_dist_with_mu = dist_with_mu
                if max_distance < min_dist_with_mu:
                    max_distance = min_dist_with_mu
                    max_dist_index = j
            mus.append(x[max_dist_index])
            mus_label.append(y[max_dist_index])
        vector_array = np.array([])
        for i in range(k):
            if i == 0:
                vector_array = mus[i]
            else:
                mus[i] = mus[i].reshape(mus[0].shape)
                vector_array = np.append(vector_array, mus[i], axis=0)
        vector_label_array = np.array(mus_label)
        return vector_array, vector_label_array
    def get_vector_index(self, x):
        min_dist_with_mu = 999999
        index = -1
        for i in range(self.vector_array.shape[0]):
            dist_with_mu = self.dist(self.vector_array[i], x)
            if min_dist_with_mu > dist_with_mu:
                min_dist_with_mu = dist_with_mu
                index = i
        return index
    def fit(self, x, y):
        self.vector_array, self.vector_label_array = self.get_vector(x, y)
        iter = 0
        while(iter < self.max_iter):
            old_vector_array = copy.deepcopy(self.vector_array)
            index = np.random.choice(y.shape[0], 1, replace=False)
            vector_index = self.get_vector_index(x[index])
            if self.vector_label_array[vector_index] == y[index]:
                self.vector_array[vector_index] = self.vector_array[vector_index] + \
                    self.eta * (x[index] - self.vector_array[vector_index])
            else:
                self.vector_array[vector_index] = self.vector_array[vector_index] - \
                    self.eta * (x[index] - self.vector_array[vector_index])
            diff = 0
            for i in range(self.vector_array.shape[0]):
                diff += np.linalg.norm(self.vector_array[i] - old_vector_array[i])
            if diff < self.e:
                print('迭代{}次退出'.format(iter))
                return
            iter += 1
        print("迭代超过{}次，退出迭代".format(self.max_iter))

# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import bench_functions as bf
import math
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['font.family'] = 'JBHGSS2'

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# plt.rcParams['figure.figsize'] = (6.0, 6.0)
class PSO_AWDV: # PSO-ADWV
    def __init__(self, sizepop=130, rangepop=(-100,100), rangespeed=(-1, 1), maxgen=900, weight=1, c1=(2.5,0.5), c2=(0.5,2.5), cycle_gen=10, func=bf.func1, dim=20):
        self.sizepop = sizepop
        self.rangepop = rangepop
        self.rangespeed = rangespeed
        self.maxgen = maxgen
        self.weight = weight
        self.result = np.zeros(maxgen)
        self.cycle_gen = cycle_gen
        self.func = func
        self.dim = dim
        self.c1 = c1
        self.c2 = c2
        self.lr = [0,0]


    def initpopvfit(self):
        pop = np.zeros((self.sizepop, self.dim))  # 初始化粒子位置
        v = np.zeros((self.sizepop, self.dim))  # 初始化粒子速度
        fitness = np.zeros(self.sizepop)  # 初始化粒子适应度值

        for i in range(self.sizepop):
            pop[i] = [(np.random.rand() - 0.5) * self.rangepop[0] * 2 for _ in range(self.dim)]
            v[i] = [(np.random.rand() - 0.5) * self.rangespeed[0] * 2 for _ in range(self.dim)]
            fitness[i] = self.func(pop[i])
        return pop, v, fitness

    def getinitbest(self, fitness, pop):
        gbestfitness = fitness.min()
        gbestpop = pop[fitness.argmin()].copy()
        pbestfitness = fitness.copy()
        pbestpop = pop.copy()
        return gbestpop, gbestfitness, pbestpop, pbestfitness

    def update(self, gen, pop, v, prev, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness):
        new_prev = v
        # 更新速度
        v = self.weight * v + (1-self.weight)*prev + self.lr[0] * np.random.rand() * (pbestpop - pop) + self.lr[1] * np.random.rand() * (
                    gbestpop - pop)
        # 更新位置
        v = np.maximum(v, self.rangespeed[0])
        v = np.minimum(v, self.rangespeed[1])

        pop = pop + v
        # 更新适应度值
        for i in range(self.sizepop):
            fitness[i] = self.func(pop[i])
        # 更新群体最优
        if fitness.min() < gbestfitness:
            gbestfitness = fitness.min()
            gbestpop = pop[fitness.argmin()].copy()
        # 更新个体最优
        for i in range(self.sizepop):
            if fitness[i] < pbestfitness[i]:
                pbestfitness[i] = fitness[i]
                pbestpop[i] = pop[i].copy()
        return pop, v, new_prev, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness

    def run(self):
        # 初始化粒子位置、速度、适应度值
        pop, v, fitness = self.initpopvfit()
        prev = v
        # 初始化群体最优、个体最优
        gbestpop, gbestfitness, pbestpop, pbestfitness = self.getinitbest(fitness, pop)
        # 迭代
        for gen in range(self.maxgen):
            E = (abs(max(fitness)) - abs(min(fitness)))/abs(max(fitness))

            self.weight = 1-0.9/(1+math.exp(0.5*E))

            self.lr[0] = (self.c1[0] - self.c1[1]) * (self.maxgen - gen) / self.maxgen + self.c1[1]
            self.lr[1] = (self.c2[0] - self.c2[1]) * (self.maxgen - gen) / self.maxgen + self.c2[1]
            # 更新粒子位置、速度、适应度值
            pop, v, prev, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness = self.update(gen,pop, v, prev, fitness, gbestpop,
                                                                                        gbestfitness, pbestpop,
                                                                                        pbestfitness)

            # 每隔10代打印一次最优值
            if gen % 10 == 0:
                print('第{}代的最优值为：{}'.format(gen, gbestfitness))
            self.result[gen] = gbestfitness

    def plot(self):
        plt.plot(self.result)
        # plt.xlabel('generation')
        # plt.ylabel('fitness')
        # plt.show()







if __name__ == '__main__':
    sizepop = 100
    # rangepop = [10, 10]
    # rangespeed = [0.1, 0.1]
    maxgen = 1000
    # weight = 0.5
    # lr = [0.5, 0.5]
    pso = PSO(sizepop=sizepop, maxgen=maxgen)
    pso.run()
    pso.plot()


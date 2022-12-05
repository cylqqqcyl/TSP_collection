# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import bench_functions as bf
import math
from mpl_toolkits.mplot3d import Axes3D
from numba.experimental import jitclass


plt.rcParams['font.family'] = 'JBHGSS2'

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.figsize'] = (10.0, 6.0)
# plt.rcParams['figure.figsize'] = (6.0, 6.0)


class MDPSO: # MDPSO
    def __init__(self, sizepop=130, rangepop=(-100,100), rangespeed=(-1, 1), maxgen=900, weight1=0.9, weight2=0.4, cycle_gen=10, func=bf.func1, dim=20):
        self.sizepop = sizepop
        self.rangepop = rangepop
        self.rangespeed = rangespeed
        self.maxgen = maxgen
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight = 1
        self.lr = [self.weight1, self.weight2]
        self.result = np.zeros(maxgen)
        self.cycle_gen = cycle_gen
        self.func = func
        self.dim = dim
        self.gbestfit_record = np.zeros((self.maxgen,self.dim))
        self.pbestfit_record = np.zeros((self.maxgen,self.sizepop,self.dim))
        self.si = 0
        self.sg = 0
        self.ti = 0
        self.tg = 0


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

    def update(self, gen, pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness):
        # 更新速度
        for i in range(self.sizepop):
            v[i] = self.weight * v[i] + self.lr[0] * np.random.rand() * (pbestpop[i] - pop[i]) + self.lr[1] * np.random.rand() * (
                        gbestpop - pop[i]) + self.si*self.lr[0] * np.random.rand() * (self.pbestfit_record[gen - self.ti,i] - pop[i])\
                +self.sg*self.lr[1] * np.random.rand() * (self.gbestfit_record[gen - self.tg] - pop[i])
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
        return pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness

    def mean_dis(self, pop):
        mean_dis = np.zeros(self.sizepop)

        for i in range(self.sizepop):
            for j in range(self.sizepop):
                mean_dis[i] += np.linalg.norm(pop[i]-pop[j])
        mean_dis = mean_dis / (self.sizepop - 1)
        return mean_dis

    def run(self):
        # 初始化粒子位置、速度、适应度值
        pop, v, fitness = self.initpopvfit()
        # 初始化群体最优、个体最优
        gbestpop, gbestfitness, pbestpop, pbestfitness = self.getinitbest(fitness, pop)
        self.gbestfit_record[0] = gbestpop
        self.pbestfit_record[0] = pbestpop
        # 迭代
        for gen in range(self.maxgen):
            mean_dis = self.mean_dis(pop)
            Ef = (mean_dis[fitness.argmin()] - mean_dis.min()) / (mean_dis.max() - mean_dis.min())
            if Ef < 0.25:
                #estate = 1
                self.si = 0
                self.sg = 0
                self.ti = 0
                self.tg = 0
            elif Ef < 0.5:
                #estate = 2
                self.si = Ef
                self.sg = 0
                self.ti = np.random.randint(max(gen,1))
                self.tg = 0
            elif Ef < 0.75:
                #estate = 3
                self.si = 0
                self.sg = Ef
                self.ti = 0
                self.tg = np.random.randint(max(gen,1))
            else:
                #estate = 4
                self.si = Ef
                self.sg = Ef
                self.ti = np.random.randint(max(gen,1))
                self.tg = np.random.randint(max(gen,1))
            self.weight = self.weight2 + (self.weight1 - self.weight2) * ((self.maxgen - gen) / self.maxgen)
            self.lr[0] = 2 * (self.maxgen - gen) / (self.maxgen) + 0.5
            self.lr[1] = - 2 * (self.maxgen - gen) / (self.maxgen) + 2.5

            # 更新粒子位置、速度、适应度值
            pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness = self.update(gen,pop, v, fitness, gbestpop,
                                                                                        gbestfitness, pbestpop,
                                                                                        pbestfitness)
            self.gbestfit_record[gen] = gbestpop
            self.pbestfit_record[gen] = pbestpop
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
    pso = MDPSO(sizepop=sizepop, maxgen=maxgen)
    pso.run()
    pso.plot()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.savefig('PSO.png')
    plt.show()


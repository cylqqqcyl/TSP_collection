# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import bench_functions as bf
import math
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'JBHGSS2'
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (10.0, 6.0)

class PSO():
    def __init__(self, sizepop=50, rangepop=(-100,100), rangespeed=(-10, 10), maxgen=300, weight=1, lr=(0.49445, 1.49445), cycle_gen=10, func=bf.func1):
        self.sizepop = sizepop
        self.rangepop = rangepop
        self.rangespeed = rangespeed
        self.maxgen = maxgen
        self.weight = weight
        self.lr = lr
        self.result = np.zeros(maxgen)
        self.cycle_gen = cycle_gen
        self.func = func


    def initpopvfit(self):
        pop = np.zeros((self.sizepop, 2))  # 初始化粒子位置
        v = np.zeros((self.sizepop, 2))  # 初始化粒子速度
        fitness = np.zeros(self.sizepop)  # 初始化粒子适应度值

        for i in range(self.sizepop):
            pop[i] = [(np.random.rand() - 0.5) * self.rangepop[0] * 2, (np.random.rand() - 0.5) * self.rangepop[1] * 2]
            v[i] = [(np.random.rand() - 0.5) * self.rangepop[0] * 2, (np.random.rand() - 0.5) * self.rangepop[1] * 2]
            fitness[i] = self.func(pop[i])
        return pop, v, fitness

    def getinitbest(self, fitness, pop):
        gbestfitness = fitness.min()
        gbestpop = pop[fitness.argmin()].copy()
        pbestfitness = fitness.copy()
        pbestpop = pop.copy()
        return gbestpop, gbestfitness, pbestpop, pbestfitness

    def life_cycle(self,pop,fitness):
        # 适应度值最大和最小的粒子位置下标
        worst_fitnessindex = fitness.argmax()
        best_fitnessindex = fitness.argmin()
        # 适应度值最大粒子换到最小的粒子位置
        pop[best_fitnessindex] = pop[worst_fitnessindex].copy()

    def social_distancing(self, pop, threshold=0.5):
        # 阈值
        threshold = np.sqrt(self.rangespeed[1])
        # 速度
        reject_v = np.zeros((self.sizepop, 2))
        # 要访问的粒子
        x = np.linspace(0, self.sizepop - 1, self.sizepop, dtype=int)
        to_visit,_ = np.meshgrid(x, x)
        for i in range(self.sizepop):
            to_visit[i, i] = -1
            for j in to_visit[i]:
                if j == -1:
                    continue
                if np.linalg.norm(pop[i] - pop[j]) < threshold:
                    new_v = pop[j] - pop[i]
                    gt_zero = np.where(new_v > 0)
                    lt_zero = np.where(new_v < 0)
                    new_v = new_v**2
                    new_v[gt_zero] = np.maximum(self.rangespeed[1]-new_v[gt_zero],0)
                    new_v[lt_zero] = np.minimum(self.rangespeed[0]-new_v[lt_zero],0)
                    reject_v[i] += new_v
                    reject_v[j] -= new_v
                to_visit[j, i] = -1
        return reject_v

    def update(self, gen, pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness):
        # 排斥速度
        reject_v = self.social_distancing(pop)
        # 更新速度
        v = self.weight * v + self.lr[0] * np.random.rand() * (pbestpop - pop) + self.lr[1] * np.random.rand() * (
                    gbestpop - pop) - reject_v
        # 更新位置
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
        if gen % self.cycle_gen == 0:
            self.life_cycle(pop, fitness)
        return pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness

    def run(self):
        # 初始化粒子位置、速度、适应度值
        pop, v, fitness = self.initpopvfit()
        # 初始化群体最优、个体最优
        gbestpop, gbestfitness, pbestpop, pbestfitness = self.getinitbest(fitness, pop)
        # 迭代
        for gen in range(self.maxgen):
            # 更新粒子位置、速度、适应度值
            pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness = self.update(gen,pop, v, fitness, gbestpop,
                                                                                        gbestfitness, pbestpop,
                                                                                        pbestfitness)
            # 每隔10代打印一次最优值
            if gen % 10 == 0:
                print('第{}代的最优值为：{}'.format(gen, gbestfitness))
            self.result[gen] = gbestfitness

    def plot(self):
        plt.plot(self.result)
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.show()

def test(psos, funcs):

    for func in funcs:
        result = []
        params = []
        for pso in psos:
            pso.func = func
            if func.__name__[-1] == '2':
                pso.rangepop = [-10, 10]
                pso.rangespeed = [-1, 1]
            elif func.__name__[-1] == '5':
                pso.rangepop = [-30, 30]
                pso.rangespeed = [-3, 3]
            pso.run()
            params.append(pso.maxgen)
            result.append(pso.result[-1])
        plt.plot(params, result, label=func.__name__)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.legend()
    # plt.savefig(u'figure/iteration.png',bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # weight = 0.5
    # lr = [0.5, 0.5]
    funcs = [bf.func1,bf.func2,bf.func3,bf.func4,bf.func5]
    pso10 = PSO(sizepop=10)
    pso100 = PSO(sizepop=100)
    pso1000 = PSO(sizepop=1000)
    pso10000 = PSO(sizepop=10000)
    psos = [pso10,pso100,pso1000,pso10000]

    test(psos, funcs)



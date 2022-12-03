# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import bench_functions as bf
import time
from mpl_toolkits.mplot3d import Axes3D
from PSO_og import PSO
from PSO_AWDV import PSO_AWDV
from MDPSO import MDPSO
from tqdm import tqdm
from numba import jit
from numba.experimental import jitclass

plt.rcParams['font.family'] = 'JBHGSS2'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.figsize'] = (10.0, 6.0)


class PSO_LCSD():
    def __init__(self, sizepop=130, rangepop=(-100,100), rangespeed=(-1, 1), maxgen=900, weight1=0.9, weight2=0.4, cycle_gen=40, func=bf.func1, dim=2):
        self.sizepop = sizepop
        self.rangepop = rangepop
        self.rangespeed = rangespeed
        self.maxgen = maxgen
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight = 1
        self.lr = [0.9, 0.4]
        self.result = np.zeros(maxgen)
        self.cycle_gen = cycle_gen
        self.func = func
        self.dim = dim
        self.count = np.zeros(sizepop)
        self.final_pop = np.zeros((sizepop, dim))


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

    def life_cycle(self,pop,fitness):
        # 密度最大和最小的粒子位置下标
        most_index = self.count.argmax()
        least_index = self.count.argmin()
        # 密度最大粒子换到最小的粒子位置
        pop[most_index] = pop[least_index].copy()

    def social_distancing(self, pop):
        # 阈值
        threshold = np.sqrt(self.rangespeed[1])
        # 速度
        reject_v = np.zeros((self.sizepop, self.dim))
        # 要访问的粒子
        x = np.linspace(0, self.sizepop - 1, self.sizepop, dtype=int)
        to_visit,_ = np.meshgrid(x, x)
        self.count = np.zeros(self.sizepop)
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
                    self.count[i] += 1
                    self.count[j] += 1
                to_visit[j, i] = -1
        return reject_v

    def update(self, gen, pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness):
        # 排斥速度
        reject_v = self.social_distancing(pop)
        # 更新速度
        v = self.weight * v + self.lr[0] * np.random.rand() * (pbestpop - pop) + self.lr[1] * np.random.rand() * (
                    gbestpop - pop) - reject_v
        v = np.maximum(v, self.rangespeed[0])
        v = np.minimum(v, self.rangespeed[1])
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
            self.weight = self.weight2 + (self.weight1 - self.weight2) * ((self.maxgen - gen) / self.maxgen)
            self.lr[0] = 2 * (self.maxgen - gen) / (self.maxgen) + 0.5
            self.lr[1] = - 2 * (self.maxgen - gen) / (self.maxgen) + 2.5
            # 更新粒子位置、速度、适应度值
            pop, v, fitness, gbestpop, gbestfitness, pbestpop, pbestfitness = self.update(gen,pop, v, fitness, gbestpop,
                                                                                        gbestfitness, pbestpop,
                                                                                        pbestfitness)
            # 每隔10代打印一次最优值
            if gen % 10 == 0:
                print('第{}代的最优值为：{}'.format(gen, gbestfitness))
            self.result[gen] = gbestfitness
        self.final_pop = pop

    def plot(self):
        plt.plot(self.result)
        # plt.xlabel('generation')
        # plt.ylabel('fitness')
        # plt.show()


def test_parameter(psos, funcs):
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
            params.append(pso.cycle_gen)
            result.append(pso.result[-1])
        plt.plot(params, result, label=func.__name__)
    plt.xlabel('生命周期')
    plt.ylabel('适应度值')
    plt.legend()
    plt.savefig(u'figure/cycle.png',bbox_inches='tight')
    plt.show()


def test_fitness(psos,func,iters=10,stride=100):
    results = np.zeros((iters,len(psos), psos[0].maxgen//stride))
    avg_time = np.zeros(len(psos))
    for iter in tqdm(range(iters)):
        for i, pso in enumerate(psos):
            t1 = time.time()
            pso.func = func
            if func.__name__[-1] == '2':
                pso.rangepop = [-10, 10]
                pso.rangespeed = [-0.1, 0.1]
            elif func.__name__[-1] == '5':
                pso.rangepop = [-30, 30]
                pso.rangespeed = [-0.3, 0.3]
            elif func.__name__[-1] == '6':
                pso.rangepop = [-5.12, 5.12]
                pso.rangespeed = [-0.0512, 0.0512]
            elif func.__name__[-1] == '7':
                pso.rangepop = [-32, 32]
                pso.rangespeed = [-0.32, 0.32]
            elif func.__name__[-1] == '8':
                pso.rangepop = [-600, 600]
                pso.rangespeed = [-6, 6]
            elif func.__name__[-1] == '9':
                pso.rangepop = [-50, 50]
                pso.rangespeed = [-0.5, 0.5]
            pso.run()
            t2 = time.time()
            avg_time[i] += t2 - t1
            for j in range(pso.maxgen//stride):
                results[iter,i,j] += pso.result[j*stride]
                # results[i, j] = np.log(pso.result[j*stride])

    for i, pso in enumerate(psos):
        # for j in range(pso.maxgen//stride):
        #     results[:,i,j] = np.log(results[:,i,j]
        #     results[i, j] = np.log(pso.result[j*stride])
        plt.plot(np.log(np.mean(results[:,i,:],axis=0)),label=pso.__class__.__name__)
    plt.xticks(range(len(psos[0].result)//stride), range(0,psos[0].maxgen,stride))
    plt.xlabel('迭代次数')
    plt.ylabel('平均适应度log(f(x))')
    plt.legend()
    plt.savefig(u'figure/{}.png'.format(func.__name__), bbox_inches='tight')
    plt.close()
    # plt.show()
    # write results to txt
    with open(u'figure/{}.txt'.format(func.__name__), 'w') as f:
        for i, pso in enumerate(psos):
            f.write(
                pso.__class__.__name__ + '\t' + 'max' + '\t' + 'min' + '\t' + 'mean' + '\t' + 'std' + '\t' + 'time' + '\n')
            f.write(str(np.max(results[:, i, -1])) + '&\t' + str(np.min(results[:, i, -1])) + '&\t' + str(
                np.mean(results[:, i, -1])) + '&\t' + str(np.std(results[:, i, -1])) +
                    '&\t' + str(avg_time[i] / iters) + '\\\\ \n')



def test_2_dim(pso, func, test_range):
    test_range = np.array(test_range)
    pso_og = PSO()
    pso_og.func = func
    pso_og.dim = 2
    pso_og.rangepop = test_range
    pso_og.rangespeed = test_range / 100
    pso_og.run()
    plt.scatter(pso_og.final_pop[:, 0], pso_og.final_pop[:, 1], label='PSO',c='skyblue', s=10,alpha=0.5)

    pso.func = func
    pso.dim = 2
    pso.rangepop = test_range
    pso.rangespeed = test_range / 100
    pso.run()
    plt.scatter(pso.final_pop[:, 0], pso.final_pop[:, 1], label=pso.__class__.__name__,c='salmon', s=10,alpha=0.5)

    contour_range = max(np.max(np.abs(pso.final_pop)),np.max(np.abs(pso_og.final_pop)))
    x = np.linspace(-contour_range, contour_range, 100)
    y = np.linspace(-contour_range, contour_range, 100)

    X, Y = np.meshgrid(x, y)

    Z = func(np.asarray([X, Y]))
    plt.contour(X, Y, Z, 10, cmap='rainbow', linewidths=0.5)
    plt.axvline(x=0, c="grey")
    plt.axhline(y=0, c="grey")
    plt.legend()
    plt.savefig(u'figure/{}_2dim.png'.format(func.__name__), bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    iters = 100
    funcs = [bf.func1,bf.func2,bf.func3,bf.func4,bf.func5]
    all_funcs = [bf.func1,bf.func2,bf.func3,bf.func4,bf.func5,bf.func6,bf.func7,bf.func8,bf.func9]
    # all_funcs = [bf.func1]

    for func in all_funcs[2:]:
        pso_og = PSO(dim=100)
        pso_awdv = PSO_AWDV(dim=100)
        mdpso = MDPSO(dim=100)
        pso_lcsd = PSO_LCSD(dim=100)
        psos = [pso_og,pso_awdv,mdpso,pso_lcsd]
        test_fitness(psos, func, iters=iters, stride=100)

    # pso_lcsd = PSO_LCSD(dim=2)
    # # test_parameter(psos, funcs)
    # for func in all_funcs[-1:]:
    #     test_range = [-100, 100]
    #     if func.__name__[-1] == '2':
    #         test_range = [-10,10]
    #     elif func.__name__[-1] == '5':
    #         test_range = [-30,30]
    #     elif func.__name__[-1] == '6':
    #         test_range = [-5.12,5.12]
    #     elif func.__name__[-1] == '7':
    #         test_range = [-32,32]
    #     elif func.__name__[-1] == '8':
    #         test_range = [-600,600]
    #     elif func.__name__[-1] == '9':
    #         test_range = [-10,10]
    #     test_2_dim(pso_lcsd, func, test_range)




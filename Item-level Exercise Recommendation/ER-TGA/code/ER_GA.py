import random
from numpy import dot
from numpy.linalg import norm
import statistics

import sys
sys.path.append("../")


class Genetic_Algorithm():
    def __init__(self, CE, S_num, C_num, X, Q, de, ds, WKC, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001,
                 Rec_num=5):
        self.E_num = [len(CE[i]) for i in range(S_num)]
        self.CE = CE
        self.S_num = S_num
        self.C_num = C_num
        self.Q = Q
        self.de = de
        self.ds = ds
        self.WKC = WKC
        self.delta = delta
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.new_offsprings_num = new_offsprings_num
        self.Rec_num = Rec_num
        self.X = X
        self.all_correct_knowledge = [set() for i in range(self.S_num)]
        for si in range(self.S_num):
            for (ej, correct) in self.X[si]:
                if correct == 0:
                    continue
                for ck in range(self.C_num):
                    if self.Q[ej][ck] == 1:
                        self.all_correct_knowledge[si].add(ck)

    def evaluate(self):
        Rel = self.forward()
        Acc = [0 for i in range(self.S_num)]
        Nov = [0 for i in range(self.S_num)]
        Div = [0 for i in range(self.S_num)]
        Proximity = [0 for i in range(self.S_num)]
        Coverage = [0 for i in range(self.S_num)]
        Quantity = [0 for i in range(self.S_num)]
        Volatility = [0 for i in range(self.S_num)]

        for si in range(self.S_num):
            if self.E_num[si] <= 0:
                continue
            if si >= self.S_num // 10:
                break
            for j in range(self.Rec_num):
                Rel_list = Rel[si][j]
                acc, nov, div, Pro, Cov, Qua, Vol = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, Rel_list)
                Acc[si] = max(Acc[si], acc)
                Nov[si] = max(Nov[si], nov)
                Div[si] = max(Div[si], div)
                Proximity[si] = max(Proximity[si], Pro)
                Coverage[si] = max(Coverage[si], Cov)
                Quantity[si] = max(Quantity[si], Qua)
                Volatility[si] = max(Volatility[si], Vol)

        print('Before deleting 0 elements in the evaluation metric list:')
        print('Acc_max=',max(Acc),'Acc_mean=', statistics.mean(Acc), 'Acc_std=', statistics.stdev(Acc), 'Acc_min=',min(Acc))
        print('Nov_max=',max(Nov),'Nov_mean=', statistics.mean(Nov), 'Nov_std=', statistics.stdev(Nov), 'Nov_min=',min(Nov))
        print('Div_max=',max(Div),'Div_mean=', statistics.mean(Div), 'Div_std=', statistics.stdev(Div), 'Div_min=',min(Div))
        print('Proximity_max=', max(Proximity), 'Proximity_mean=', statistics.mean(Proximity), 'Proximity_std=', statistics.stdev(Proximity), 'Proximity_min=',min(Proximity))
        print('Coverage_max=', max(Coverage), 'Coverage_mean=', statistics.mean(Coverage), 'Coverage_std=',statistics.stdev(Coverage), 'Coverage_min=', min(Coverage))
        print('Quantity_max=', max(Quantity), 'Quantity_mean=', statistics.mean(Quantity), 'Quantity_std=',statistics.stdev(Quantity), 'Quantity_min=', min(Quantity))
        print('Volatility_max=', max(Volatility), 'Volatility_mean=', statistics.mean(Volatility), 'Volatility_std=',statistics.stdev(Volatility), 'Volatility_min=', min(Volatility))
        Acc, Nov, Div = [x for x in Acc if x > 0], [x for x in Nov if x > 0], [x for x in Div if x > 0]
        Proximity, Coverage, Quantity, Volatility = [x for x in Proximity if x > 0], [x for x in Coverage if x > 0], [x for x in Quantity if x > 0], [x for x in Volatility if x > 0]
        print('After deleting 0 elements in the evaluation metric list:')
        print('Acc_max=',max(Acc),'Acc_mean=', statistics.mean(Acc), 'Acc_std=', statistics.stdev(Acc), 'Acc_min=',min(Acc))
        print('Nov_max=',max(Nov),'Nov_mean=', statistics.mean(Nov), 'Nov_std=', statistics.stdev(Nov), 'Nov_min=',min(Nov))
        print('Div_max=',max(Div),'Div_mean=', statistics.mean(Div), 'Div_std=', statistics.stdev(Div), 'Div_min=',min(Div))
        print('Proximity_max=', max(Proximity), 'Proximity_mean=', statistics.mean(Proximity), 'Proximity_std=',statistics.stdev(Proximity), 'Proximity_min=', min(Proximity))
        print('Coverage_max=', max(Coverage), 'Coverage_mean=', statistics.mean(Coverage), 'Coverage_std=',statistics.stdev(Coverage), 'Coverage_min=', min(Coverage))
        print('Quantity_max=', max(Quantity), 'Quantity_mean=', statistics.mean(Quantity), 'Quantity_std=',statistics.stdev(Quantity), 'Quantity_min=', min(Quantity))
        print('Volatility_max=', max(Volatility), 'Volatility_mean=', statistics.mean(Volatility), 'Volatility_std=',statistics.stdev(Volatility), 'Volatility_min=', min(Volatility))
    def forward(self):
        Rel = [[] for i in range(self.S_num)]
        for si in range(self.S_num):
            if self.E_num[si] <= 0:
                continue
            if si >= self.S_num // 10:
                break
            population = self.Init_Population(si)
            for age in range(self.generation_num):
                print(age, end='  ')
                population = self.Crossover(si, population)
                population = self.Selection(si, population)
            for c in range(self.Rec_num):
                ls = []
                chromosome = population[c]
                for k in range(len(chromosome)):
                    if chromosome[k] == 0:
                        continue
                    ls.append(self.CE[si][k])
                print(ls)
                Rel[si].append(ls)
        return Rel
    def Init_Population(self, si):
        population = [[random.randint(0, 1) for i in range(self.E_num[si])] for j in range(self.population_size)]
        return population

    def Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(self, si, Rel_list):
        Acc, Nov, Div, Proximity, Coverage, Quantity, Volatility = 0, 0, 0, 0, 0, 0, 0

        WKC_tmp = self.WKC
        for ej in Rel_list:
            Acc += abs(self.ds[si] - self.de[ej])

            Rel_knowledge = set()

            for ck in range(self.C_num):
                if self.Q[ej][ck] == 0:
                    continue
                Rel_knowledge.add(ck)
                WKC_tmp[si][ck] = 0
            intersection = len(Rel_knowledge.intersection(self.all_correct_knowledge[si]))
            union = len(Rel_knowledge.union(self.all_correct_knowledge[si]))
            Nov += ((intersection / float(union)) if union != 0 else 0)

            for ei in Rel_list:
                if ei == ej:
                    continue
                Div += self.Sim(self.Q[ei], self.Q[ej])

            Proximity += self.de[ej]

        for j in range(len(Rel_list) - 1):
            Volatility += abs(self.de[Rel_list[j]] - self.de[Rel_list[j + 1]])

        Acc = Acc / len(Rel_list) if len(Rel_list) > 0 else 0
        Nov = Nov / len(Rel_list) if len(Rel_list) > 0 else 0
        Div = Div / (len(Rel_list) * (len(Rel_list) - 1)) if len(Rel_list) > 1 else 0
        Proximity = abs((Proximity / len(Rel_list) if len(Rel_list) > 0 else 0) - self.ds[si])
        Coverage = sum(WKC_tmp[si]) / sum(self.WKC[si]) if sum(self.WKC[si]) > 0 else 0
        Quantity = len(Rel_list) / sum(self.CE[si]) if sum(self.CE[si]) > 0 else 0
        Volatility = Volatility / len(Rel_list) if len(Rel_list) > 0 else 0

        return (1-Acc), (1-Nov), (1-Div), Proximity, Coverage, Quantity, Volatility

    def Fitness(self, si, population):
        ans = [[0, 0, 0] for i in range(len(population))]
        for c in range(self.population_size):
            chromosome = population[c]
            Rel_list = []
            for i in range(len(chromosome)):
                if chromosome[i] == 0:
                    continue
                ei = self.CE[si][i]
                Rel_list.append(ei)

            ans[c][0], ans[c][1], ans[c][2], ans[c][3], ans[c][4], ans[c][5], ans[c][6] = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, Rel_list)
        fitness = [sum(ans[c]) for c in range(len(population))]
        return fitness
    def Selection(self, si, population):
        fitness = self.Fitness(si, population)
        print([round(x, 3) for x in fitness])
        sorted_pairs = sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)
        sorted_fitness, sorted_population = zip(*sorted_pairs)
        sorted_population = list(sorted_population)
        offsprings = sorted_population[:self.population_size]
        return offsprings
    def Crossover(self, si, population):
        offsprings = population
        cnt = 0
        num = len(population)
        while cnt <= (int)(self.E_num[si] * self.new_offsprings_num):
            c1, c2 = random.randint(0, num - 1), random.randint(0, num - 1)
            partial = random.randint(0, self.E_num[si]-1)
            offsprings.append(population[c1][:partial] + population[c2][partial:])
            cnt += 1
        return offsprings
    def Mutation(self, si, population):
        offsprings = population
        cnt = 0
        num = len(population)
        while cnt <= (int)(self.E_num[si] * self.new_offsprings_num):
            c = random.randint(0, num - 1)
            partial = random.randint(0, self.E_num[si] - 1)
            chromosome = population[c]
            chromosome[partial] ^= 1
            offsprings.append(chromosome)
            cnt += 1
        return offsprings
    def Sim(self, qi, qj):
        a = norm(qi)
        b = norm(qj)
        return dot(qi, qj) / (a * b) if a * b != 0 else 0
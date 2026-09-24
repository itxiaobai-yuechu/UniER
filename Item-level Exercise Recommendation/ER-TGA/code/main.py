import random
import numpy as np
import time
import pandas as pd
import ER_TGA
import sys
import os

import sys
import Process_Data

class My_model():
    def __init__(self, Epsilon, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001, pa=0.3, Rec_num=5, N=5):
        self.Epsilon = Epsilon
        self.delta = delta
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.pa = pa
        self.new_offsprings_num = new_offsprings_num
        self.Rec_num = Rec_num
        self.N = N

    def forward(self, name):
        my_DataSet = Process_Data.MyDataset(self.Epsilon, self.delta, self.population_size, self.generation_num, self.new_offsprings_num,
                                               self.pc, self.pm, self.Rec_num)
        if name == "Bridge2006":
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.Bridge2006()
        elif name == 'Nips34':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.Nips34()
        elif name == 'Junyi':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.Junyi()
        elif name == 'ASSISTments2009':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.ASSISTments2009()
        elif name == 'ASSISTments2012':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.ASSISTments2012()
        elif name == 'ASSISTments2017':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.ASSISTments2017()
        elif name == 'Algebra2005':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.Algebra2005()
        elif name == 'Ednet':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.Ednet()   
        elif name == 'Xes3g5m':
            self.S_num, self.E_num, self.C_num, self.Q, self.X, self.CS, self.de, self.ds, self.WKC, self.CE, problem_id_map, user_id_map = my_DataSet.Xes3g5m()   
        GA = ER_TGA.Genetic_Algorithm(CE=self.CE, S_num=self.S_num, C_num=self.C_num, X=self.X, Q=self.Q,
                                                 de=self.de, ds=self.ds, WKC=self.WKC,
                                                 delta=self.delta, population_size=self.population_size,
                                                 generation_num=self.generation_num,
                                                 new_offsprings_num=self.new_offsprings_num, pc=self.pc, pm=self.pm, pa = self.pa,
                                                 Rec_num=self.Rec_num, N=self.N, problem_id_map = problem_id_map, user_id_map = user_id_map,final_output_K = 20)

        GA.evaluate()

if __name__ == "__main__":
    seed = int(os.environ.get("UNIER_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)

    stime = time.time()
    name = sys.argv[1]
    generation_num = int(sys.argv[2])
    N =int(sys.argv[3])
    model=My_model(Epsilon=0.5, delta=0.5, population_size=50,
                    generation_num=generation_num, new_offsprings_num=0.3, pc=0.7, pm=0.0001, pa=0.3, Rec_num=1, N=N)

    model.forward(name)
    print('The running time is',time.time() - stime)

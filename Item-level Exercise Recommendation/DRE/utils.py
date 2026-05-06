import numpy as np
  
class MultiObjectiveReward:
    def __init__(self, alpha1, alpha2, alpha3, beta1, beta2, N, g):

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.beta1 = beta1
        self.beta2 = beta2
        self.N = N
        self.g = g
    
    def r1(self, performance, concept, concept_next, concept_batch_set):


        if concept_next != {} and performance == 0 and concept_next.isdisjoint(concept):
            return self.beta1                                                            
        elif concept_next != {} and concept_next.difference(concept_batch_set): 
            return self.beta2                                                   
        else:
            return 0

    def r2(self, dt, dt_next):

        if dt_next != None:
            return - (dt_next - dt) ** 2
        else:
            return 0

    def r3(self, performance_history):

        if performance_history == []:
            return 0
        else:
            phi_u_N = np.mean(performance_history[-self.N:])
            return 1 - abs(self.g - phi_u_N)
    
    def merge_reward(self, performance, concept, concept_next, dt, dt_next, performance_history, concept_batch_set):

        r1 = self.r1(performance, concept, concept_next,concept_batch_set)
        r2 = self.r2(dt, dt_next)
        r3 = self.r3(performance_history)
        r = self.alpha1 * r1 + self.alpha2 * r2 + self.alpha3 * r3
        return r

def dcg_at_k(r, k):

    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):

    ideal_r = sorted(r, reverse=True)
    dcg_max = dcg_at_k(ideal_r, k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r, k):

    relevant_at_k = sum(r[:k])

    total_relevant = sum(r)

    if total_relevant == 0:
        return 0.0

    return relevant_at_k / total_relevant


def precision_at_k(r, k):

    relevant_at_k = sum(r[:k])

    return relevant_at_k / k


def f1_at_k(r, k):

    precision = precision_at_k(r, k)
    recall = recall_at_k(r, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)




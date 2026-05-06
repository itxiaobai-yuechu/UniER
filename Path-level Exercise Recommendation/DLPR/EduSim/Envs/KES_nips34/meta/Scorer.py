import random
import numpy as np
from EduSim.Envs.meta import TraitScorer



class KESScorer(TraitScorer):
    def __init__(self, binary_scorer=True):
        super(KESScorer, self).__init__()
        self._binary = binary_scorer

    def response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if user_trait[item_trait] >= 0.5 else 0

    def middle_response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if random.random() <= user_trait[item_trait] else 0

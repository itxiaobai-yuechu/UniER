from EduSim.Envs.meta import TraitScorer
import random


class KESScorer(TraitScorer):
    def __init__(self):
        super(KESScorer, self).__init__()

    def response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if user_trait[item_trait] >= 0.5 else 0

    def middle_response_function(self, user_trait, item_trait, *args, **kwargs):
        return 1 if random.random() <= user_trait[item_trait] else 0

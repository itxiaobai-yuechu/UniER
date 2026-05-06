

import uuid
import numpy as np

__all__ = ["MetaLearner", "MetaLearnerGroup", "MetaLearningModel", "MetaFiniteLearnerGroup", "MetaInfinityLearnerGroup"]


class MetaLearningModel:
    def step(self, state, learning_item, *args, **kwargs):
        raise NotImplementedError


class MetaLearner:
    def __init__(self, user_id=None):
        self.id = self.__id(user_id)

    @classmethod
    def __id(cls, _id=None):
        return _id if _id is not None else uuid.uuid1()

    @property
    def profile(self):
        return {"id": self.id}

    @property
    def state(self):
        raise NotImplementedError

    def learn(self, learning_item, *args, **kwargs) -> ...:
        """learn a new learning item, which can result in state changing"""
        raise NotImplementedError

    def response(self, test_item, *args, **kwargs) -> ...:
        """
        give the response to the test_item

        Parameters
        ----------
        test_item
            a test item
        """
        raise NotImplementedError


class MetaLearnerGroup:
    pass


class MetaFiniteLearnerGroup(MetaLearnerGroup):
    def __init__(self, learners, seed=None):
        self._learners = learners
        self._random_state = np.random.RandomState(seed)

    def __getitem__(self, item):
        return self._learners[item]

    def sample(self):
        return self._random_state.choice(self._learners)

    def __len__(self):
        return len(self._learners)


class MetaInfinityLearnerGroup(MetaLearnerGroup):
    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        return self


import pickle
import time
import os
import networkx as nx
import numpy as np
import torch

from EduSim.Envs.KES_ASSIST17.meta.Learner import LearnerGroup, Learner
from EduSim.spaces import ListSpace
from EduSim.deep_model import KTnet
from EduSim.Envs.meta import Item
from EduSim.Envs.meta import Env
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.utils import get_proj_path, get_graph_embeddings

from .meta import KESASSISTScorer
__all__ = ["KESASSIST17Env"]


class KESASSIST17Env(Env):
    def __init__(self, dataRec_path, seed=None):
        super().__init__()
        self.target = 'all'
        self.type = 'KES'
        self.env_name = 'KESassist17'
        self.random_state = np.random.RandomState(seed)
        self.graph_embeddings = get_graph_embeddings('KES_ASSIST17')
        self.dataRec_path = dataRec_path

        self.knowledge_structure = nx.DiGraph()
        with open(f"{get_proj_path()}/data/dataProcess/assist17/nxgraph.pkl", "rb") as file:
            self.knowledge_structure = pickle.loads(file.read())
        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        assert not list(nx.algorithms.simple_cycles(self.knowledge_structure)), "loop in DiGraph"

        self.num_skills = 102
        self.max_sequence_length = 300
        self.feature_dim = 2 * self.num_skills
        self.embed_dim = 128
        self.hidden_size = 256
        self.item_list = [i for i in range(self.num_skills)]
        self.learning_item_base = [Item(item_id=i, knowledge=i) for i in self.item_list]

        dkt_para_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.01,
        }
        self.KTnet = KTnet(dkt_para_dict)
        directory = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST17/meta_data'
        dkt_file = f'{directory}/env_weights/ValBest.ckpt'
        if os.path.exists(dkt_file):
            param_dict = torch.load(dkt_file, weights_only=False)
            self.KTnet.load_state_dict(param_dict)
        else:
            raise ValueError('dkt net not trained yet!')
        self.scorer = KESASSISTScorer()
        self.action_space = ListSpace(self.item_list, seed=seed)

        self.learners = LearnerGroup(self.dataRec_path, seed=seed)
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()

    @property
    def parameters(self) -> dict:
        return {
            "action_space": self.action_space
        }

    def learn_and_test(self, learner: Learner, item_id):
        state = learner.state
        score = self.scorer.response_function(state, item_id)
        learner.learn(item_id, score)
        self.update_learner_state()
        return item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        state = learner.state
        knowledge_response = {}
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, self.scorer.response_function(state, test_item)]
        if detailed:
            return_thing = knowledge_response
        elif reduce == "sum":
            return_thing = np.sum([v for _, v in knowledge_response.values()])
        elif reduce in {"mean", "ave"}:
            return_thing = np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)
        return return_thing

    def update_learner_state(self):
        logs = self._learner.profile['logs']
        sequence_length = len(logs)
        input_data = self.get_feature_matrix(logs).unsqueeze(0)
        self._learner._state = torch.sigmoid(
            self.KTnet(input_data).permute(1, 0, 2).squeeze(0)[sequence_length - 1])

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        if self.target_type == 'all':
            self._learner._target = set(range(self.num_skills))
        self.update_learner_state()
        self._initial_score = self._exam(self._learner)
        while self._initial_score >= len(self._learner.target):
            self._learner = next(self.learners)
            if self.target_type == 'all':
                self._learner._target = set(range(self.num_skills))
            self.update_learner_state()
            self._initial_score = self._exam(self._learner)
        return self._learner.profile, self._exam(self._learner, detailed=True)

    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        self.episode_end_time = time.time()
        return observation, reward, done, info

    def step(self, learning_item_id, *args, **kwargs):
        a = self._exam(self._learner)
        observation = self.learn_and_test(self._learner, learning_item_id)
        b = self._exam(self._learner)
        return observation, b - a, b == len(self._learner.target), None

    def n_step(self, learning_path, *args, **kwargs):
        exercise_history = []
        a = self._exam(self._learner)
        for learning_item_id in learning_path:
            item_id, score = self.learn_and_test(self._learner, learning_item_id)
            exercise_history.append([item_id, score])
        b = self._exam(self._learner)
        return exercise_history, b - a, b == len(self._learner.target), None

    def reset(self):
        self._learner = None

    def render(self, mode='human'):
        if mode == "log":
            return_thing = "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )
        else:
            return_thing = 'for else return'
        return return_thing

    def get_feature_matrix(self, session):
        input_data = np.zeros(shape=(max(1, len(session)), self.feature_dim), dtype=np.float32)
        j = 0
        while j < len(session):
            problem_id = session[j][0]
            if session[j][1] == 0:
                input_data[j][problem_id] = 1.0
            elif session[j][1] == 1:
                input_data[j][problem_id + self.num_skills] = 1.0
            j += 1
        return torch.tensor(input_data)

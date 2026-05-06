from copy import deepcopy
import networkx as nx
import random
from EduSim.Envs.meta import Env

import numpy as np
from EduSim.Envs.KES_junyi.meta.Learner import LearnerGroup, Learner
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.spaces import ListSpace
from .meta import KESItemBase, KESScorer
from .utils import load_environment_parameters
import math
import torch
import os

from EduSim.deep_model import KTnet
import sys
sys.path.append('../')
from KT import Agent_KT

def get_kt_mastery(kt_net, concepts, answers, num_concepts, feature_dim, device):
    with torch.no_grad():
        kt_input = torch.zeros((1, len(concepts), feature_dim)).to(device)
        for i, (c, a) in enumerate(zip(concepts, answers)):
            kt_input[0, i, int(c) + (num_concepts if a == 1 else 0)] = 1
        if len(concepts) > 0:
            kt_output = kt_net(kt_input)[-1, 0, :]
            kt_output = torch.sigmoid(kt_output)
        else:
            kt_output = torch.zeros(num_concepts).to(device)
        return kt_output

def load_kt_net(num_concepts, device, directory):
    feature_dim = 2 * num_concepts
    embed_dim = 128
    hidden_size = 256
    dkt_para_dict = {
        'input_size': feature_dim,
        'emb_dim': embed_dim,
        'hidden_size': hidden_size,
        'num_skills': num_concepts,
        'nlayers': 2,
        'dropout': 0.01,
    }
    kt_net = KTnet(dkt_para_dict).to(device)
    dkt_file = f'{directory}/env_weights/ValBest.ckpt'
    if os.path.exists(dkt_file):
        param_dict = torch.load(dkt_file, map_location=device)
        kt_net.load_state_dict(param_dict)
        kt_net.eval()
    else:
        raise ValueError('dkt net not trained yet!')
    return kt_net, feature_dim



__all__ = ["KESjunyiEnv"]


class KESjunyiEnv(Env):
    def __init__(self, dataRec_path, dataRec_q_path, seed=None):
        self.random_state = np.random.RandomState(seed)
        self.dataRec_path = dataRec_path
        self.dataRec_q_path = dataRec_q_path

        parameters = load_environment_parameters()
        self.knowledge_structure = parameters["knowledge_structure"]
        self.know_item = parameters["know_item"]
        self.concept_difficulty = parameters["concept_difficulty"]
        
        self._item_base = KESItemBase(
            parameters["knowledge_structure"],
            parameters["learning_order"],
            items=parameters["items"]
        )
        self.learning_item_base = deepcopy(self._item_base)
        self.learning_item_base.drop_attribute()
        self.test_item_base = self._item_base
        self.scorer = KESScorer(parameters["configuration"].get("binary_scorer", True))

        self.action_space = ListSpace(self.learning_item_base.know_id_list, seed=seed)
        self.learners = LearnerGroup(self.dataRec_path, self.dataRec_q_path, seed=seed)

        self._order_ratio = parameters["configuration"]["order_ratio"]
        self._review_times = parameters["configuration"]["review_times"]
        self._learning_order = parameters["learning_order"]

        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        self._initial_step = parameters["configuration"]["initial_steps"]

        self._learner = None
        self._initial_score = None
        self._exam_reduce = "sum" if parameters["configuration"].get("exam_sum", True) else "ave"

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_concepts = 39
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_data")
        self.kt_net, self.feature_dim = load_kt_net(self.num_concepts, self.device, directory)

        self.KT_model = Agent_KT('junyi', self.num_concepts, 721, self.learning_item_base ,self.concept_difficulty)
        self.mastery_threshold = 0.5

        self.target_type = 'portion'

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
            "learning_item_base": self.learning_item_base
        }

    def learn_and_test(self, learner: Learner, practice_item_id, pre_mastery, is_commit=False):
        learning_item_id = self.learning_item_base.index[str(practice_item_id)].knowledge
        if pre_mastery > self.mastery_threshold:
            score = 1
        else:
            score = self.scorer.response_function(self._learner._state, practice_item_id)
        
        if is_commit:
            learner.learn(learning_item_id, practice_item_id, score)
            self.update_learner_state()
            
        return learning_item_id, practice_item_id, score

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        logs = learner.profile['logs']
        concepts = [int(log[0]) for log in logs]
        answers = [int(log[1]) for log in logs]
        state = get_kt_mastery(self.kt_net, concepts, answers, self.num_concepts, self.feature_dim, self.device)
        
        knowledge_response = {}
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, 1 if state[test_item].item() > 0.5 else 0]
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
        logs_q = self._learner.profile['logs_q']
        init_concepts = []
        init_answers = []
        init_questions = []
        for log in logs:
            init_concepts.append(int(log[0]))
            init_answers.append(int(log[1]))
        for log in logs_q:
            init_questions.append(int(log[0]))
            
        questions_difficulty_H = [self.learning_item_base.index[str(qid)].difficulty * 50 for qid in init_questions]
        concepts_difficulty_H = [self.concept_difficulty[str(c)] * 50 for c in init_concepts]
        self._learner._state =  self.KT_model.forward_state(init_questions, init_concepts, questions_difficulty_H, concepts_difficulty_H, init_answers).squeeze(0)

    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners) 
        if self.target_type == 'all':
            self._learner._target = set(range(self.num_concepts))
        self.update_learner_state()
        self._initial_score = self._exam(self._learner)
        while self._initial_score >= len(self._learner.target):
            self._learner = next(self.learners)
            if self.target_type == 'all':
                self._learner._target = set(range(self.num_concepts))
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
        return observation, reward, done, info

    def step_l(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score = self._initial_score
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        print(info)
        return observation, reward, done, info

    def step_p(self, current_questions, current_concepts, current_answers, practice_item_id, is_commit=False, *args, **kwargs):
        concept = self.learning_item_base.index[str(practice_item_id)].knowledge
        pre_diff = self.learning_item_base.index[str(current_questions[-1])].difficulty
        learning_item_id = self.learning_item_base.index[str(practice_item_id)].knowledge
        pre_kt_mastery = get_kt_mastery(self.kt_net, current_concepts, current_answers, self.num_concepts, self.feature_dim, self.device)
        pre_mastery = pre_kt_mastery[learning_item_id].item()
        observation = self.learn_and_test(self._learner, practice_item_id, pre_mastery, is_commit)
        curr_state = self._learner._state
        cur_diff = self.learning_item_base.index[str(practice_item_id)].difficulty
        if is_commit:
            current_questions = current_questions + [practice_item_id]
            current_concepts = current_concepts + [concept]
            current_answers = current_answers + [observation[2]]
        curr_kt_mastery = get_kt_mastery(self.kt_net, current_concepts, current_answers, self.num_concepts, self.feature_dim, self.device)
        curr_mastery = curr_kt_mastery[learning_item_id].item()
        done = curr_mastery > self.mastery_threshold
        reward_smooth = -math.pow((cur_diff - pre_diff), 5)
        reward_improve = curr_mastery - pre_mastery
        p_reward = 0.5 * reward_smooth + 0.5 * reward_improve
        return curr_state.unsqueeze(0), observation, p_reward, done, cur_diff , current_questions, current_concepts,current_answers
    
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
            return "target: %s, state: %s" % (
                self._learner.target, dict(self._exam(self._learner))
            )
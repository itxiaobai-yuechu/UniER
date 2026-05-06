

from collections import defaultdict
from longling.ML.toolkit.monitor import EMAValue
import numpy as np
from tqdm import tqdm
import os
from EduSim.utils import get_proj_path

STEP = 10
EPISODE = 20
SUMMARY = 30

str2level = {
    "step": STEP,
    "episode": EPISODE,
    "summary": SUMMARY,
}


def as_level(obj):
    if isinstance(obj, int):
        return_thing = obj
    else:
        return_thing = str2level[obj]
    return return_thing


def meta_train_eval(env_input_dict):

    agent = env_input_dict['agent']
    env = env_input_dict['env']
    max_steps = env_input_dict['max_steps']
    max_episode_num = env_input_dict['max_episode_num']
    n_step = env_input_dict['n_step']
    train = env_input_dict['train']
    logger = env_input_dict['logger']
    values = env_input_dict['values']
    level = env_input_dict['level']

    episode = 0

    level = as_level(level)

    rewards = []
    infos = []

    if values is None:
        values = {"Episode": EMAValue(["Reward"])}

    loop = max_episode_num
    logs_ = defaultdict(lambda: {})

    for i in range(loop):
        if max_episode_num is not None and episode >= max_episode_num:
            break
        try:
            learner_profile = env.begin_episode()
            agent.begin_episode(learner_profile)
            episode += 1
            
            print(f"\n========== Episode: {episode} / {max_episode_num} ==========")

            if level <= as_level("episode"):
                logger.info("episode [%s]: %s" % (episode, env.render("log")))

        except StopIteration:
            break

        step_ = 0
        logs_[i * 3][0] = step_

        if n_step is True:
            assert max_steps is not None
            learning_path = agent.n_step(max_steps)
            for observation, reward, done, info in env.n_step(learning_path):
                agent.observe(observation, reward, done, info)
                if done:
                    break
        else:
            learning_path = []
            if max_steps is not None:
                for _ in range(max_steps):
                    try:
                        learning_item = agent.step()
                        learning_path.append(learning_item)
                    except StopIteration:
                        break
                    observation, reward, done, info = env.step(learning_item)
                    agent.observe(observation, reward, done, info)
                    if done:
                        break
            else:
                raise ValueError("max_steps should be set when n_step is False")
        observation, reward, done, info = env.end_episode()
        if not info:
            info = {}
        
        print(f"Episode {episode} Result -> episode_reward: {reward}")
        
        agent.end_episode(observation, reward, done, info)
        rewards.append(reward)
        infos.append(info)
        values["Episode"].update("Reward", reward)
        env.reset()


    print("================ START TESTING ================")
    test_learner_nums = [1000]
    num_mean_rewards = []

    for test_num in test_learner_nums:
        test_rewards = []
        for i in tqdm(range(test_num), desc=f"Testing {test_num} learners"):
            learner_profile = env.begin_episode()
            agent.begin_episode(learner_profile)
            
            if max_steps is not None:
                for _ in range(max_steps):
                    try:
                        learning_item = agent.step()
                    except StopIteration:
                        break
                    
                    observation, reward, done, info = env.step(learning_item)
                    agent.observe(observation, reward, done, info)
                    if done:
                        break
            
            observation, reward, done, info = env.end_episode()
            env.reset()  
            test_rewards.append(reward)

        mean_reward = np.mean(np.array(test_rewards))
        num_mean_rewards.append(mean_reward)
        
    num_mean_rewards = np.array(num_mean_rewards)
    print(f'\nAgent test mean rewards on learner nums: {test_learner_nums} rewards: {num_mean_rewards}')

    print("===============================================\n")

def train_eval(env_input_dict):
    max_episode_num = env_input_dict['max_episode_num']
    assert max_episode_num is not None, "infinity environment, max_episode_num should be set"

    meta_train_eval(env_input_dict=env_input_dict)

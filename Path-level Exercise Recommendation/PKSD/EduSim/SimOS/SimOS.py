


from itertools import cycle
from gym.spaces import Space
import logging
from collections import defaultdict
import pandas as pd

from EduSim.Envs.meta.Env import Env
from EduSim.utils.callback import get_board_episode_callback, reward_summary_callback
from longling.ML.toolkit.monitor import ConsoleProgressMonitor, EMAValue

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
        return obj
    else:
        return str2level[obj]


def meta_train_eval(agent,
                    env: Env,
                    max_steps: int = None,
                    max_episode_num: int = None,
                    n_step=False,
                    train=False,
                    logger=logging,
                    level="episode",
                    episode_callback=None,
                    summary_callback=None,
                    values: dict = None,
                    monitor=None,
                    sw=None):

    episode = 0

    level = as_level(level)

    rewards = []
    infos = []

    if values is None:
        values = {}

    if monitor is None:        
        monitor = lambda x: x

    loop = cycle([1]) if max_episode_num is None else range(max_episode_num)

    logs_ = defaultdict(lambda: {})

    for i in monitor(loop):        
        if max_episode_num is not None and episode >= max_episode_num:
            break
        try:
            learner_profile = env.begin_episode()
            agent.begin_episode(learner_profile)
            episode += 1
            if level <= as_level("episode"):
                logger.info("episode [%s]: %s" % (episode, env.render("log")))

        except StopIteration:
            break

        step_ = 0
        logs_[i * 3][0] = step_

        episode_answers = []

        if n_step is True:
            assert max_steps is not None
            learning_path = agent.n_step(max_steps)
            for observation, reward, done, info in env.n_step(learning_path):
                episode_answers.append(str(int(observation[1])))
                agent.observe(observation, reward, done, info)
                if done:
                    break
        else:
            learning_path = []
            _step = 0
            if max_steps is not None:
                for j in range(max_steps):
                    try:
                        learning_item = agent.step()
                        learning_path.append(learning_item)
                    except StopIteration:
                        break
                    observation, reward, done, info = env.step(learning_item)
                    episode_answers.append(str(int(observation[1])))


                    if level <= as_level("step"):
                        _step += 1
                        logger.debug(
                            "step [%s]: agent -|%s|-> env, env state %s" % (_step, learning_item, env.render("log"))
                        )
                        logger.debug(
                            "step [%s]: observation: %s, reward: %s" % (_step, observation, reward)
                        )
                    agent.observe(observation, reward, done, info)
                    if done:
                        break
            else:
                while True:
                    learning_item = agent.step()
                    observation, reward, done, info = env.step(learning_item)
                    learning_path.append(learning_item)
                    episode_answers.append(str(int(observation[1])))
                    agent.observe(observation, reward, done, info)
                    if done:
                        break
        observation, reward, done, info = env.end_episode()
        if not info:
            info = {}
        info['record'] = ""
        agent.end_episode(observation, reward, done, info)
        rewards.append(reward)
        infos.append(info)

        if level <= as_level("episode"):
            logger.info("episode [%s] - learning path: %s" % (episode, learning_path))
            logger.info("episode [%s] - total reward: %s" % (episode, reward))
            logger.info("episode [%s]: %s" % (episode, env.render(mode="log")))
        
        print(f"========== Episode: {episode} / {max_episode_num} ==========")
        print(f"Episode {episode} Result -> episode_reward: {reward}\n")
        
        path_str_list = [str(x) for x in learning_path]
        print(f"(['{len(learning_path)}'], {path_str_list}, {episode_answers})")
        print(f"reward: {reward}")
        
        init_score = info.get('initial_score', 'N/A')
        final_score = info.get('final_score', 'N/A')
        print(f"info{{'initial_score': {init_score}, 'final_score': {final_score}}}")

        if sw is not None:
            sw.add_scalar('Episode-Reward:', reward, episode)

        env.reset()

        if episode_callback is not None:
            episode_callback(episode, reward, done, info, logger)

        if train is True:
            agent.tune()


    if summary_callback is not None and level <= as_level("summary"):
        return summary_callback(rewards, infos, logger)


def train_eval(agent, env: Env, max_steps: int = None, max_episode_num: int = None, n_step=False,
               train=False,
               logger=logging, level="episode", board_dir=None,
               sw=None, episode_callback=None, summary_callback=None,
               *args, **kwargs):


    assert max_episode_num is not None, "infinity environment, max_episode_num should be set"

    if episode_callback is None:
        sw, episode_callback = get_board_episode_callback(board_dir, sw=sw)

    if summary_callback is None:
        summary_callback = reward_summary_callback

    meta_train_eval(
        agent, env,
        max_steps, max_episode_num, n_step, train,
        logger, level,
        episode_callback=episode_callback,
        summary_callback=summary_callback,
        sw=sw,
        *args, **kwargs
    )

    if board_dir:
        sw.close()





def episode_reward(initial_score, final_score, full_score) -> (int, float):
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score

    return delta / normalize_factor

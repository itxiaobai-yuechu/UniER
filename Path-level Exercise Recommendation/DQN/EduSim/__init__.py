


from gym.envs.registration import register
from .Envs import *
from .SimOS import train_eval
from .spaces import *

register(
    id='KES-v1',
    entry_point='EduSim.Envs:KESASSIST09Env'
)

register(
    id='KES-v2',
    entry_point='EduSim.Envs:KESASSIST12Env'
)

register(
    id='KES-v3',
    entry_point='EduSim.Envs:KESASSIST15Env'
)

register(
    id='KES-v4',
    entry_point='EduSim.Envs:KESASSIST17Env'
)

register(
    id='KES-v5',
    entry_point='EduSim.Envs:KESalgebra2005Env'
)

register(
    id='KES-v6',
    entry_point='EduSim.Envs:KESbridge2006Env'
)

register(
    id='KES-v7',
    entry_point='EduSim.Envs:KESednetEnv'
)

register(
    id='KES-v8',
    entry_point='EduSim.Envs:KESjunyiEnv'
)

register(
    id='KES-v9',
    entry_point='EduSim.Envs:KESnips34Env'
)

register(
    id='KES-v10',
    entry_point='EduSim.Envs:KESxes3g5mEnv'
)

register(
    id='KES-v11',
    entry_point='EduSim.Envs:KESmooccubexEnv'
)
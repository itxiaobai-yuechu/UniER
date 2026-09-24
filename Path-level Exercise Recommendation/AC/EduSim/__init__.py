


from gym.envs.registration import register
from .Envs import *
from .SimOS import train_eval
from .spaces import *

register(
    id='KES-v1',
    entry_point='EduSim.Envs:KESEnv'
)
register(
    id='KESASSIST15-v1',
    entry_point='EduSim.Envs:KESASSIST15Env'
)
register(
    id='KSS-v2',
    entry_point='EduSim.Envs:KSSEnv',
)
register(
    id='KESASSIST09-v1',
    entry_point='EduSim.Envs:KESASSIST09Env'
)
register(
    id='KESASSIST12-v1',
    entry_point='EduSim.Envs:KESASSIST12Env'
)
register(
    id='KESASSIST17-v1',
    entry_point='EduSim.Envs:KESASSIST17Env'
)
register(
    id='KESalgebra2005-v1',
    entry_point='EduSim.Envs:KESalgebra2005Env'
)
register(
    id='KESbridge2006-v1',
    entry_point='EduSim.Envs:KESbridge2006Env'
)
register(
    id='KESednet-v1',
    entry_point='EduSim.Envs:KESednetEnv'
)
register(
    id='KESnips34-v1',
    entry_point='EduSim.Envs:KESnips34Env'
)
register(
    id='KESjunyi-v1',
    entry_point='EduSim.Envs:KESjunyiEnv'
)
register(
    id='KESmooccube-v1',
    entry_point='EduSim.Envs:KESmooccubeEnv'
)
register(
    id='KESxes3g5m-v1',
    entry_point='EduSim.Envs:KESxes3g5mEnv'
)
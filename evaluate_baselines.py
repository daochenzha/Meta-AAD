from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from meta_aad.agents import RandomAgent, IForestAgent
from meta_aad.utils import evaluate
from meta_aad.env import EvalEnv

DATAPATH = "data/annthyroid.csv"

if __name__ == "__main__":

    # Random Agent
    eval_env = EvalEnv(datapath=DATAPATH, budget=100)
    agent = RandomAgent()
    reward, results = evaluate(eval_env, agent)
    print("Random Agent")
    print('-----------------------')
    print('Reward: {}'.format(reward))
    print('Curve: {}'.format(' '.join([str(_r) for _r in results])))
    print('')

    # IForest Agent
    eval_env = EvalEnv(datapath=DATAPATH, budget=100)
    agent = IForestAgent(DATAPATH)
    reward, results = evaluate(eval_env, agent)
    print("Random Agent")
    print('-----------------------')
    print('Reward: {}'.format(reward))
    print('Curve: {}'.format(' '.join([str(_r) for _r in results])))
    print('')
    

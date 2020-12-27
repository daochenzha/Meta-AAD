import numpy as np
from sklearn.preprocessing import StandardScaler, normalize

from meta_aad.utils import read_data_as_matrix, rank_scores, remove_illegal, run_iforest

class BaseAgent(object):
    def __init__(self):
        pass

    def eval_step(self, s, legal):
        """ Perform one step in active anomaly detection,
            i.e., choose one instance for query from the
            remaining unlabeled instances

        Args:
            s: the current state
            legal: the legal actions (unlabled instances)

        Returns:
            The index of the instance to query
        """
        pass
        
class RandomAgent(BaseAgent):
    def eval_step(self, s, legal):
        return np.random.choice(legal)

class IForestAgent(BaseAgent):
    def __init__(self, datapath):
        X_train, labels, anomalies = read_data_as_matrix(datapath)
        X_train = StandardScaler().fit_transform(X_train)
        self.scores = run_iforest(X_train)
        self.scores = np.exp(-self.scores)

    def eval_step(self, s, legal):
        rank = rank_scores(remove_illegal(self.scores, legal))
        return rank[-1] 

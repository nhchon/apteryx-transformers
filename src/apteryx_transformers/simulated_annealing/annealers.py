import numpy as np
from tqdm import tqdm

class Annealer:
    def __init__(self, scorer, proposer):
        self.scorer = scorer
        self.proposer = proposer

    def propose_until_accepted(self, y, T):
        its = 1
        while True:
            candidate = self.proposer.propose(y)
            p_accept = min(1, (np.exp((self.scorer.score(candidate['output']) - self.scorer.score(y)) / T)))
            accepted = (np.random.uniform(0, 1) < p_accept)
            if accepted:
                return candidate['output'], its
            its += 1

    def anneal(self,
               y_init,
               max_search=1000,
               T_init=10,
               C=.1,
               eps=1 / 1e10):
        y = y_init
        y_hist = [y]
        t_hist = [T_init]
        it_hist = [0]
        for t in tqdm(range(max_search)):
            T = max(T_init - (C * t), eps)
            y, its = self.propose_until_accepted(y, T)
            y_hist.append(y)
            t_hist.append(T)
            it_hist.append(its)

        return {'y': y,
                'y_hist': y_hist,
                't_hist': t_hist,
                'it_hist': it_hist}

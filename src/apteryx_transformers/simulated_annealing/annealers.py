import numpy as np

class Annealer:
    def __init__(self, scoring_fn, proposal_fn, proposal_set):
        self.scoring_fn = scoring_fn
        self.proposal_fn = proposal_fn
        self.proposal_set = proposal_set

    def propose_until_accepted(self, y, T):
        its = 1
        while True:
            candidate = self.proposal_fn(self.proposal_set)
            p_accept = min(1, (np.exp((self.scoring_fn(candidate) - self.scoring_fn(y)) / T)))
            accepted = (np.random.uniform(0, 1) < p_accept)
            if accepted:
                return (candidate, its)
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
        for t in range(max_search):
            T = max(T_init - (C * t), eps)
            y, its = self.propose_until_accepted(y, T)
            y_hist.append(y)
            t_hist.append(T)
            it_hist.append(its)

        return {'y': y,
                'y_hist': y_hist,
                't_hist': t_hist,
                'it_hist': it_hist}

import numpy as np
from tqdm import tqdm
import time


class Annealer:
    def __init__(
        self,
        scorer,
        proposer,
        score_hyperparameters=dict(),
        proposer_hyperparameters=dict(),
    ):
        # Score output is {'total': num, 'constituent_1': num, ...'constituent_N':num}
        self.scorer = scorer

        self.proposer = proposer

        self.score_hyperparameters = score_hyperparameters
        self.proposer_hyperparameters = proposer_hyperparameters

    def propose_until_accepted(self, y, T):
        its = 1
        while True:
            proposal = self.proposer.propose(y, **self.proposer_hyperparameters)
            candidate_seq = proposal["output"]
            candidate_score = self.scorer.score(
                candidate_seq, **self.score_hyperparameters
            )
            y_score = self.scorer.score(y, **self.score_hyperparameters)
            p_accept = min(
                1, (np.exp((candidate_score["total"] - y_score["total"]) / T))
            )
            accepted = np.random.uniform(0, 1) < p_accept
            if accepted:
                return candidate_seq, its, candidate_score, y_score
            its += 1

    def anneal(
        self,
        y_init,
        max_search=1000,
        T_init=10,
        C=0.1,
        eps=1 / 1e10,
        log=False,
        log_fn=lambda x: print(x),
        time_limit: float = 60,
    ):
        y = y_init
        y_hist = [y]
        t_hist = [T_init]
        it_hist = [0]
        candidate_score_hist = []
        y_score_hist = []

        start_time = time.time()
        for t in tqdm(range(max_search)):
            if log:
                log_fn(y)
            current_time = time.time() - start_time
            if (y == "") or current_time > time_limit:
                print(
                    f"""Breaking Early.
                    compute time: {current_time}
                    y: {y}"""
                )
                # Break early if converged to empty string or overrun time limit.
                break
            T = max(T_init - (C * t), eps)
            y, its, candidate_score, y_score = self.propose_until_accepted(y, T)
            y_hist.append(y)
            t_hist.append(T)
            it_hist.append(its)
            candidate_score_hist.append(candidate_score)
            y_score_hist.append(y_score)

        return {
            "y": y,
            "y_hist": y_hist,
            "t_hist": t_hist,
            "it_hist": it_hist,
            "y_score_hist": y_score_hist,
            "candidate_score_hist": candidate_score_hist,
        }

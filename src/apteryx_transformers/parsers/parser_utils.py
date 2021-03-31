from fuzzysearch import find_near_matches
from fuzzywuzzy import process
import string
import pandas as pd
import numpy as np
import json


def fuzzy_extract(qs, ls, threshold):
    '''fuzzy matches 'qs' in 'ls' and returns list of
    tuples of (word,index)
    '''
    for word, _ in process.extractBests(qs, (ls,), score_cutoff=threshold):
        for match in find_near_matches(qs, word, max_l_dist=1):
            match = word[match.start:match.end]
            index = ls.find(match)
            yield (match, index)


def remove_tables(s, severity=0):
    n_ws = []
    n_alpha = []
    n_num = []
    n_total = []
    lines = []
    for line in s.split('\n'):
        n_alpha.append(sum([c in string.ascii_letters for c in line]))
        n_ws.append(sum([c in string.whitespace for c in line]))
        n_num.append(sum([c in string.digits for c in line]))
        n_total.append(len(line))
        lines.append(line)

    df = pd.DataFrame.from_records(zip(n_ws, n_alpha, n_num, n_total, lines))
    df.columns = ['ws', 'alpha', 'num', 'total', 'line']

    # anti-tableness heuristic: rewards dense alpha, penalize dense whitespace/numbers
    df['table_score'] = np.log(((df.alpha + 1) / (df.ws + df.num + 1)) * (df.total / len(df) + 1))

    return '\n'.join(df[df.table_score > severity].line.values)


def serialize_report(d):
    new_d = dict()
    for k,v in d.items():
        if isinstance(v, dict):
            new_d[k] = serialize_report(v)
        else:
            new_d[k] = v.to_json(orient = 'records')
    return json.dumps(new_d)

def is_json(s):
    try:
        _ = json.loads(s)
        return True
    except:
        return False

def deserialize_nested(s):
    if isinstance(s, dict):
        return {k:deserialize_nested(v) for k,v in s.items()}
    elif isinstance(s, list):
        return [deserialize_nested(i) for i in s]
    elif isinstance(s, str):
        if is_json(s):
            return deserialize_nested(json.loads(s))
        else:
            return s
    else:
        return s
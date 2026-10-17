import hashlib
import numpy as np


def select_round(train,rng,budget=4975):
    assert len(train)>=budget
    return rng.permutation(len(train))[:budget]


def order_hash(rows):
    return hashlib.sha256('\n'.join(r['root_id'] for r in rows).encode()).hexdigest()

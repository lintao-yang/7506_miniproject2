"""Train the candidate verifier ensemble and export a NumPy-only weight bundle.

Usage (from code/):
    python train.py --output checkpoint

Only the training split and its references are used. The exported bundle holds one
scaler + weights per ensemble member, so inference never needs scikit-learn.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import load_inputs, parse_answer, read_jsonl
from submission import BLOCKS, build_features, design

# (kind, feature mask over ('emb','center','text','sim'), regularisation)
MEMBERS = [
    ('lr', (1, 0, 1, 1), 0.003),
    ('lr', (1, 0, 1, 1), 0.01),
    ('lr', (0, 1, 1, 1), 0.01),
    ('mlp', (0, 1, 1, 0), 1e-3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('checkpoint'))
    ap.add_argument('--hidden', type=int, default=48)
    args = ap.parse_args()

    bank, arrays = load_inputs('train')
    gold = read_jsonl(Path('data') / 'gsm8k_train.jsonl')
    labels = np.array([[int(parse_answer(c['text']) is not None
                            and parse_answer(c['text']) == parse_answer(g['answer']))
                        for c in row['candidates']] for row, g in zip(bank, gold)])
    # Trivial groups (all four candidates correct) carry no ranking signal.
    keep = labels.sum(axis=1) < 4
    print('train questions=%d  with >=1 wrong candidate=%d' % (len(bank), keep.sum()))

    feats = build_features(bank, arrays['embeddings'])
    y = labels[keep].reshape(-1)
    bundle = {'n_models': np.array([len(MEMBERS)]),
              'use': np.array([m[1] for m in MEMBERS], dtype=np.int64),
              'kind': np.array([0 if m[0] == 'lr' else 1 for m in MEMBERS], dtype=np.int64)}
    for m, (kind, use, c) in enumerate(MEMBERS):
        x = design(feats, use)[np.repeat(keep, 4)]
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        if kind == 'lr':
            model = LogisticRegression(C=c, max_iter=2000).fit(z, y)
            bundle['coef%d' % m] = model.coef_.reshape(-1)
            bundle['b%d' % m] = np.array([model.intercept_[0]])
        else:
            model = MLPClassifier(hidden_layer_sizes=(args.hidden,), alpha=c, max_iter=400,
                                  random_state=0, early_stopping=True).fit(z, y)
            bundle['W1_%d' % m] = model.coefs_[0]
            bundle['b1_%d' % m] = model.intercepts_[0]
            bundle['W2_%d' % m] = model.coefs_[1].reshape(-1, 1)
            bundle['b2_%d' % m] = model.intercepts_[1]
        bundle['mean%d' % m] = scaler.mean_
        bundle['scale%d' % m] = scaler.scale_
        print('member %d: %s blocks=%s dim=%d' % (m, kind, [b for b, u in zip(BLOCKS, use) if u], x.shape[1]))

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'verifier_weights.npz'
    np.savez(target, **bundle)
    print('wrote %s (%.1f KiB)' % (target, target.stat().st_size / 1024))


if __name__ == '__main__':
    main()

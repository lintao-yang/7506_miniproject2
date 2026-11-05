"""Ablation and stratified analysis for the submitted verifier.

Usage (from code/):
    python experiments.py --output results/ablation.json

Trains one model per feature configuration on the training split (groups with at
least one wrong candidate) and reports accuracy on validation and test. Also
reports the shipped ensemble and its accuracy by number of distinct parsed
answers. Test references are used for reporting only, never for fitting.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import load_inputs, parse_answer, read_jsonl
from submission import build_features, design, predict

C_TEXT = 0.01


def load_split(split):
    bank, arrays = load_inputs(split)
    gold = read_jsonl(Path('data') / ('gsm8k_%s.jsonl' % split))
    labels = np.array([[int(parse_answer(c['text']) is not None
                            and parse_answer(c['text']) == parse_answer(g['answer']))
                        for c in row['candidates']] for row, g in zip(bank, gold)])
    feats = build_features(bank, arrays['embeddings'])
    feats = {k: v.astype(np.float32) for k, v in feats.items()}
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    nuniq = np.array([len({a for a in ans if a is not None}) for ans in answers])
    return {'bank': bank, 'arrays': arrays, 'y': labels, 'feats': feats,
            'answers': answers, 'nuniq': nuniq}


def vote(answers):
    out = []
    for ans in answers:
        cnt = collections.Counter(a for a in ans if a is not None)
        top = max(cnt.values()) if cnt else 0
        winners = [i for i, a in enumerate(ans) if a is not None and cnt[a] == top]
        out.append(winners[0] if winners else 0)
    return np.array(out)


def accuracy(choices, y):
    return 100.0 * y[np.arange(len(y)), choices].mean()


def fit(tr, cfg):
    use, c, text_cols = cfg
    mask = tr['y'].sum(axis=1) < 4
    parts = []
    for flag, name in zip(use, ('emb', 'center', 'text', 'sim')):
        if flag:
            block = tr['feats'][name]
            if name == 'text' and text_cols is not None:
                block = block[:, :, text_cols]
            parts.append(block.reshape(-1, block.shape[-1]))
    x = np.concatenate(parts, axis=1)[np.repeat(mask, 4)]
    y = tr['y'][mask].reshape(-1)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(x), y)
    return clf, scaler, use, text_cols


def apply_model(model, sp):
    clf, scaler, use, text_cols = model
    parts = []
    for flag, name in zip(use, ('emb', 'center', 'text', 'sim')):
        if flag:
            block = sp['feats'][name]
            if name == 'text' and text_cols is not None:
                block = block[:, :, text_cols]
            parts.append(block.reshape(-1, block.shape[-1]))
    p = clf.predict_proba(scaler.transform(np.concatenate(parts, axis=1)))[:, 1]
    return p.reshape(-1, 4).argmax(axis=1)


STRUCT = tuple(range(0, 10))
FORMAT = (10, 11, 12, 15, 16)
ANSWER = (13, 14, 17)

CONFIGS = [
    ('index 0 (no method)', None),
    ('majority vote', None),
    ('emb only', ((1, 0, 0, 0), 0.003, None)),
    ('center only', ((0, 1, 0, 0), 0.003, None)),
    ('sim only', ((0, 0, 0, 1), 0.01, None)),
    ('text only (all)', ((0, 0, 1, 0), C_TEXT, None)),
    ('text: structure+arithmetic', ((0, 0, 1, 0), C_TEXT, STRUCT)),
    ('text: format markers', ((0, 0, 1, 0), C_TEXT, FORMAT)),
    ('text: answer plausibility', ((0, 0, 1, 0), C_TEXT, ANSWER)),
    ('emb+text', ((1, 0, 1, 0), 0.003, None)),
    ('emb+text+sim', ((1, 0, 1, 1), 0.01, None)),
    ('center+text+sim', ((0, 1, 1, 1), 0.01, None)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('results/ablation.json'))
    ap.add_argument('--checkpoint', type=Path, default=Path('checkpoint'))
    args = ap.parse_args()

    tr = load_split('train')
    va = load_split('validation')
    te = load_split('test')
    print('train=%d (ambiguous %d)  val=%d (ambiguous %d)  test=%d (ambiguous %d)' % (
        len(tr['y']), (tr['y'].sum(1) < 4).sum(), len(va['y']), (va['nuniq'] >= 2).sum(),
        len(te['y']), (te['nuniq'] >= 2).sum()))

    out = {'protocol': 'dase7506-mp2-open-hard-verifier-v5',
           'note': 'trained on train groups with >=1 wrong candidate; test used for reporting only',
           'baselines': {}, 'ablations': {}, 'final_ensemble': {}}
    rows = []

    for name, cfg in CONFIGS:
        if cfg is None:
            if name == 'index 0 (no method)':
                choices = {s: np.zeros(len(d['y']), dtype=int) for s, d in
                           (('validation', va), ('test', te))}
            else:
                choices = {s: vote(d['answers']) for s, d in (('validation', va), ('test', te))}
        else:
            model = fit(tr, cfg)
            choices = {s: apply_model(model, d) for s, d in (('validation', va), ('test', te))}
        rec = {}
        for split, d in (('validation', va), ('test', te)):
            acc = accuracy(choices[split], d['y'])
            amb = d['nuniq'] >= 2
            acc_amb = 100.0 * d['y'][np.arange(len(d['y'])), choices[split]][amb].mean() if amb.any() else None
            rec[split] = {'accuracy': round(acc, 2), 'accuracy_ambiguous': round(acc_amb, 2) if acc_amb else None,
                          'correct': int(d['y'][np.arange(len(d['y'])), choices[split]].sum()), 'n': len(d['y'])}
        rows.append((name, rec['validation']['accuracy'], rec['validation']['accuracy_ambiguous'],
                     rec['test']['accuracy']))
        (out['baselines'] if cfg is None else out['ablations'])[name] = rec

    # shipped ensemble, evaluated through the exported checkpoint
    for split, d in (('validation', va), ('test', te)):
        choices = np.array(predict(d['bank'], d['arrays'], args.checkpoint))
        acc = accuracy(choices, d['y'])
        strata = {}
        for k in (1, 2, 3, 4):
            m = d['nuniq'] == k
            if m.any():
                hit = d['y'][np.arange(len(d['y'])), choices][m]
                strata['nuniq=%d' % k] = {'n': int(m.sum()), 'accuracy': round(100.0 * hit.mean(), 2)}
        out['final_ensemble'][split] = {'accuracy': round(acc, 2),
                                        'correct': int(d['y'][np.arange(len(d['y'])), choices].sum()),
                                        'n': len(d['y']), 'by_distinct_answers': strata}

    print('\n%-28s %10s %10s %10s' % ('configuration', 'val full', 'val amb', 'test'))
    for r in rows:
        print('%-28s %10.1f %10.1f %10.1f' % r)
    print('\nshipped ensemble: val=%.2f test=%.2f' % (
        out['final_ensemble']['validation']['accuracy'], out['final_ensemble']['test']['accuracy']))
    for split in ('validation', 'test'):
        print('  %s by distinct answers: %s' % (split, json.dumps(out['final_ensemble'][split]['by_distinct_answers'])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print('\nwrote', args.output)


if __name__ == '__main__':
    main()

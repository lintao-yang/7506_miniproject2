"""MP2 submission: verify each candidate and pick the most likely correct one.

Method: candidate-level feature set (frozen trace embedding, embedding centred on
the four-candidate group, cheap text/arity statistics, cross-candidate similarity)
scored by a small ensemble of linear probes plus one compact MLP. Every parameter
is a plain NumPy array stored in the checkpoint; inference is a few matrix products.
"""
import re
from pathlib import Path

import numpy as np

NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
EQ_RE = re.compile(
    r'([\d.,]+(?:\s*[/]\s*[\d.,]+)?)\s*([+\-*/xX\u00d7\u00f7])\s*'
    r'([\d.,]+(?:\s*[/]\s*[\d.,]+)?)\s*=\s*([\d.,]+)')
WORD_NUM = {w: i for i, w in enumerate(
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
     'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
     'seventeen', 'eighteen', 'nineteen', 'twenty'])}

BLOCKS = ('emb', 'center', 'text', 'sim')
N_TEXT = 18
N_SIM = 3


def parse_answer(text):
    """Same numeric-answer rule as the official scorer."""
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    match = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
    if not match:
        return None
    from fractions import Fraction
    try:
        return str(Fraction(match.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def _numval(s):
    if s is None:
        return None
    try:
        if '/' in s:
            a, b = s.split('/')
            return float(a) / float(b)
        return float(s.replace(',', ''))
    except Exception:
        return None


def _numbers(text):
    out = []
    for m in NUM_RE.finditer(text):
        v = _numval(m.group().replace(',', ''))
        if v is not None:
            out.append(v)
    return out


def text_features(question, text, ans):
    """18 cheap statistics: length/format, question-number coverage, arithmetic self-consistency."""
    av = _numval(ans)
    qnums = _numbers(question)
    qwords = {w for w in re.findall(r'[a-zA-Z]+', question.lower()) if w in WORD_NUM}
    cn = _numbers(text)
    cov = [int(any(abs(v - x) < 1e-9 for x in cn)) for v in qnums]
    wcov = [int(w in text.lower()) for w in qwords]
    checked = errs = 0
    for m in EQ_RE.finditer(text):
        a = _numval(m.group(1).replace(' ', ''))
        b = _numval(m.group(3).replace(' ', ''))
        c = _numval(m.group(4).replace(' ', ''))
        op = m.group(2)
        if a is None or b is None or c is None:
            continue
        if op == '-':
            v = a - b
        elif op == '+':
            v = a + b
        elif op in 'xX*\u00d7':
            v = a * b
        elif op in '/\u00f7':
            if b == 0:
                continue
            v = a / b
        else:
            continue
        checked += 1
        errs += int(abs(v - c) > 1e-3 * max(1.0, abs(c)))
    return np.array([
        np.log1p(len(text)) / 10, np.log1p(len(text.split())) / 10,
        text.count('\n') / 10, text.count('=') / 10,
        len(cn) / 20,
        float(np.mean(cov)) if cov else 0.0, float(np.min(cov)) if cov else 0.0,
        float(np.mean(wcov)) if wcov else 0.0,
        errs / float(max(1, checked)), checked / 10,
        float('boxed' in text), float('####' in text),
        float('**Answer' in text or 'Final Answer' in text),
        float(av is not None), np.log1p(abs(av)) / 5 if av is not None else 0.0,
        text.count('$') / 20,
        len(set(round(x, 6) for x in cn)) / 20,
        float(av is not None and any(abs(av - x) < 1e-9 for x in cn)),
    ], dtype=np.float64)


def build_features(bank, embeddings):
    """Return {block_name: float array [N, 4, dim]} for every question in bank."""
    emb = np.asarray(embeddings, dtype=np.float64).reshape(len(bank), 4, -1)
    centered = emb - emb.mean(axis=1, keepdims=True)
    unit = emb / (np.linalg.norm(emb, axis=2, keepdims=True) + 1e-9)
    sim = np.zeros((len(bank), 4, N_SIM), dtype=np.float64)
    for i in range(4):
        others = [j for j in range(4) if j != i]
        sim[:, i, 0] = (unit[:, i:i + 1, :] * unit[:, others, :]).sum(axis=2).mean(axis=1)
        sim[:, i, 1] = (unit[:, i:i + 1, :] * unit[:, others, :]).sum(axis=2).max(axis=1)
        sim[:, i, 2] = np.linalg.norm(emb[:, i, :], axis=1) / 100.0
    text = np.zeros((len(bank), 4, N_TEXT), dtype=np.float64)
    for q, row in enumerate(bank):
        for c, cand in enumerate(row['candidates']):
            text[q, c] = text_features(row['question'], cand['text'], parse_answer(cand['text']))
    return {'emb': emb, 'center': centered, 'text': text, 'sim': sim}


def design(feats, use):
    """Stack the selected feature blocks into a candidate-level design matrix."""
    parts = []
    for flag, name in zip(use, BLOCKS):
        if flag:
            parts.append(feats[name].reshape(-1, feats[name].shape[-1]))
    return np.concatenate(parts, axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def predict(bank, arrays, checkpoint):
    """Return one integer in 0..3 per question, in bank order."""
    if checkpoint is None:
        raise ValueError('This method needs --checkpoint (the trained weight bundle).')
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'verifier_weights.npz'
    with np.load(path, allow_pickle=False) as f:
        n_models = int(f['n_models'][0])
        mask = f['use'][:]
        kind = f['kind'][:]
        params = {k: f[k] for k in f.files if k not in ('n_models', 'use', 'kind')}
    feats = build_features(bank, arrays['embeddings'])
    n = len(bank)
    prob = np.zeros((n, 4), dtype=np.float64)
    for m in range(n_models):
        x = design(feats, mask[m])
        z = (x - params['mean%d' % m]) / params['scale%d' % m]
        if kind[m] == 0:
            s = _sigmoid(z @ params['coef%d' % m] + params['b%d' % m])
        else:
            h = np.maximum(z @ params['W1_%d' % m] + params['b1_%d' % m], 0.0)
            s = _sigmoid(h @ params['W2_%d' % m] + params['b2_%d' % m])
        prob += s.reshape(n, 4)
    prob /= float(n_models)
    return [int(i) for i in prob.argmax(axis=1)]

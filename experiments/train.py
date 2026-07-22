"""Train the windowed conditional model (all radii jointly) on tinyshakespeare."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time, sys
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from model import CFG, init_params, forward, save

MASK = 0
RADII = [1, 2, 4, 8, 16]          # window w = 2r+1 in {3,5,9,17,33}
STEPS = 6000
BATCH = 64
LR, WARM = 1e-3, 200

train_ids = np.load("data/train_ids.npy")
val_ids = np.load("data/val_ids.npy")

def batch_windows(ids, r, n, rng):
    w = 2 * r + 1
    starts = rng.integers(0, len(ids) - w, size=n)
    idx = starts[:, None] + np.arange(w)[None, :]
    x = ids[idx].copy()
    y = x[:, r].copy()
    x[:, r] = MASK
    return x, y

def loss_fn(params, x, y):
    logits = forward(params, x)[:, x.shape[1] // 2, :]
    logp = jax.nn.log_softmax(logits)
    return -jnp.take_along_axis(logp, y[:, None], 1).mean()

@partial(jax.jit, static_argnums=(4,))
def train_step(params, m, v, t, w, x, y):
    loss, g = jax.value_and_grad(loss_fn)(params, x, y)
    lr = LR * jnp.minimum(1.0, t / WARM) * (0.1 ** (t / STEPS))  # decay to ~0.1x
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh)
    return params, m, v, loss

@partial(jax.jit, static_argnums=(3,))
def eval_step(params, x, y, w):
    logits = forward(params, x)[:, x.shape[1] // 2, :]
    acc = (logits.argmax(-1) == y).mean()
    logp = jax.nn.log_softmax(logits)
    ce = -jnp.take_along_axis(logp, y[:, None], 1).mean()
    return acc, ce

def main(data_dir="data", ckpt_dir="ckpt", vocab=None, steps=None):
    """Train the windowed conditional model. Defaults reproduce the word-level
    pilot (vocab=2000, data/, ckpt/). Pass vocab/data_dir/ckpt_dir to train a
    BPE model (Phase 2). Only init_params depends on vocab; forward infers the
    output width from the param shapes, so the arch (d,h,layers) is unchanged."""
    import os
    global STEPS
    if steps is not None:
        STEPS = steps
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg = CFG if vocab is None else {**CFG, "vocab": vocab}
    tr = np.load(f"{data_dir}/train_ids.npy")
    va = np.load(f"{data_dir}/val_ids.npy")
    rng = np.random.default_rng(0)
    params = init_params(jax.random.PRNGKey(0), cfg)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    m, v = zeros, jax.tree_util.tree_map(jnp.zeros_like, params)
    t0 = time.time()
    for t in range(1, STEPS + 1):
        r = RADII[t % len(RADII)]
        x, y = batch_windows(tr, r, BATCH, rng)
        params, m, v, loss = train_step(params, m, v, t, 2 * r + 1,
                                        jnp.asarray(x), jnp.asarray(y))
        if t % 250 == 0 or t == 1:
            msgs = []
            for rr in RADII:
                xe, ye = batch_windows(va, rr, 512, rng)
                acc, ce = eval_step(params, jnp.asarray(xe), jnp.asarray(ye), 2 * rr + 1)
                msgs.append(f"r{rr}:acc={float(acc):.3f}/ce={float(ce):.2f}")
            el = time.time() - t0
            print(f"step {t:5d}  loss={float(loss):.3f}  [{el:6.0f}s]  " + " ".join(msgs), flush=True)
        if t % 1000 == 0 or t == STEPS:
            save(params, f"{ckpt_dir}/step{t}.npz")
    save(params, f"{ckpt_dir}/final.npz")
    print("done", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    main(a.data_dir, a.ckpt_dir, a.vocab, a.steps)

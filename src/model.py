r"""Tiny bidirectional transformer trained as a windowed conditional model:
input = window of w=2r+1 tokens with the center masked; predict the center.
This is exactly the CA rule p_r(x_i | x_{i-r..i+r \ i}) for each radius r."""
import jax, jax.numpy as jnp
import numpy as np

CFG = dict(d=96, h=4, layers=2, vocab=2000, max_w=33)

def init_params(key, cfg=CFG):
    d, V, W = cfg["d"], cfg["vocab"], cfg["max_w"]
    ks = jax.random.split(key, 20)
    s = 0.02
    def N(k, shape): return s * jax.random.normal(k, shape, jnp.float32)
    params = {
        "tok": N(ks[0], (V, d)),
        "pos": N(ks[1], (W, d)),
        "lnf": {"g": jnp.ones(d), "b": jnp.zeros(d)},
        "out_b": jnp.zeros(V),
        "blocks": [],
    }
    for i in range(cfg["layers"]):
        k = jax.random.split(ks[2 + i], 8)
        params["blocks"].append({
            "ln1": {"g": jnp.ones(d), "b": jnp.zeros(d)},
            "wq": N(k[0], (d, d)), "wk": N(k[1], (d, d)),
            "wv": N(k[2], (d, d)), "wo": N(k[3], (d, d)),
            "ln2": {"g": jnp.ones(d), "b": jnp.zeros(d)},
            "w1": N(k[4], (d, 4 * d)), "b1": jnp.zeros(4 * d),
            "w2": N(k[5], (4 * d, d)), "b2": jnp.zeros(d),
        })
    return params

def _ln(x, p):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * p["g"] + p["b"]

def forward(params, tokens, cfg=CFG):
    """tokens: (B, w) int32 -> logits (B, w, V). Bidirectional."""
    B, w = tokens.shape
    d, h = cfg["d"], cfg["h"]
    dh = d // h
    x = params["tok"][tokens] + params["pos"][:w]
    for blk in params["blocks"]:
        y = _ln(x, blk["ln1"])
        q = (y @ blk["wq"]).reshape(B, w, h, dh).transpose(0, 2, 1, 3)
        k = (y @ blk["wk"]).reshape(B, w, h, dh).transpose(0, 2, 1, 3)
        v = (y @ blk["wv"]).reshape(B, w, h, dh).transpose(0, 2, 1, 3)
        att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(dh), axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, w, d)
        x = x + o @ blk["wo"]
        y = _ln(x, blk["ln2"])
        x = x + jax.nn.gelu(y @ blk["w1"] + blk["b1"]) @ blk["w2"] + blk["b2"]
    x = _ln(x, params["lnf"])
    return x @ params["tok"].T + params["out_b"]

def center_logits(params, tokens, cfg=CFG):
    """Logits at the center position of odd-length windows: (B, V)."""
    w = tokens.shape[1]
    return forward(params, tokens, cfg)[:, w // 2, :]

def save(params, path):
    flat = jax.tree_util.tree_leaves(params)
    treedef = jax.tree_util.tree_structure(params)
    np.savez(path, *[np.asarray(x) for x in flat])
    return treedef

def load(path, cfg=CFG):
    key = jax.random.PRNGKey(0)
    ref = init_params(key, cfg)
    treedef = jax.tree_util.tree_structure(ref)
    z = np.load(path)
    leaves = [jnp.asarray(z[f]) for f in z.files]
    return jax.tree_util.tree_unflatten(treedef, leaves)

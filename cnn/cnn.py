import jax
import jax.numpy as jnp
from jax import jit
import struct


def read_label(file):
    with open(file, "rb") as f:
        struct.unpack(">II", f.read(8))
        return jnp.frombuffer(f.read(), dtype=jnp.uint8)


def read_image(file):
    with open(file, "rb") as f:
        struct.unpack(">IIII", f.read(16))
        return jnp.frombuffer(f.read(), dtype=jnp.uint8)


def relu(x):
    return jnp.maximum(0, x)


def adam(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    new_m, new_v, new_params = [], [], []
    for i, (param, grad) in enumerate(zip(params, grads)):
        m = b1 * state["m"][i] + (1 - b1) * grad
        v = b2 * state["v"][i] + (1 - b2) * grad**2
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        new_m.append(m)
        new_v.append(v)
        new_params.append(param - lr * mhat / (jnp.sqrt(vhat) + eps))
    new_state = {"m": new_m, "v": new_v, "t": t}
    return new_params, new_state


def conv2d(x, W, b):
    # Optimized by Opus 4.5
    # x:   (bs, c_in, h, w)
    # w:   (c_out, c_in, kh, kw)
    # b:   (c_out,)
    # out: (bs, c_out, h_out, w_out)

    bs, c_in, h, w = x.shape
    c_out, _, kh, kw = W.shape
    h_out = h - kh + 1
    w_out = w - kw + 1

    i_out = jnp.arange(h_out)
    j_out = jnp.arange(w_out)
    di = jnp.arange(kh)
    dj = jnp.arange(kw)

    i_idx = (i_out[:, None] + di[None, :]).ravel()
    j_idx = (j_out[:, None] + dj[None, :]).ravel()

    patches = x[:, :, i_idx[:, None], j_idx[None, :]]
    patches = patches.reshape(bs, c_in, h_out, kh, w_out, kw)
    patches = patches.transpose(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(bs, h_out * w_out, c_in * kh * kw)

    W_flat = W.reshape(c_out, -1)
    out = jnp.dot(patches, W_flat.T) + b

    return out.reshape(bs, h_out, w_out, c_out).transpose(0, 3, 1, 2)


def max_pool2d(x, k):
    # x:   (bs, c, h, w)
    # out: (bs, c, h_out, w_out)
    assert x.ndim == 4
    bs, c, h, w = x.shape

    h_out, w_out = h // k, w // k

    x_trunc = x[:, :, : h_out * k, : w_out * k]  # (bs, c, h', w')
    x_reshaped = x_trunc.reshape(bs, c, h_out, k, w_out, k)
    return x_reshaped.max(axis=(3, 5))


def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def forward(params, x, key, training=True):
    w1, b1, w2, b2, w3, b3 = params

    z1 = conv2d(x, w1, b1)  # out: (bs, 32, 26, 26)
    a1 = relu(z1)
    p1 = max_pool2d(a1, 2)  # out: (bs, 32, 13, 13)

    z2 = conv2d(p1, w2, b2)  # out: (bs, 64, 11, 11)
    a2 = relu(z2)
    p2 = max_pool2d(a2, 2)  # out: (bs, 64, 5, 5)

    # dropout
    if training:
        mask = jax.random.bernoulli(key, 0.5, p2.shape)
        p2 = p2 * mask * 2

    flattened = p2.reshape(p2.shape[0], -1)  # out: (bs, 1600)
    logits = jnp.dot(flattened, w3) + b3  # out: (bs, 10)
    return logits


def loss_fn(params, x, y, key):
    logits = forward(params, x, key)
    probs = softmax(logits)
    y_one_hot = jnp.eye(10)[y]
    loss = -jnp.mean(jnp.sum(y_one_hot * jnp.log(probs + 1e-9), axis=1))
    return loss


@jit
def step(params, x, y, key, lr, state):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, key)
    params, state = adam(params, grads, state, lr)
    return params, loss, state


@jit
def predict(params, x):
    logits = forward(params, x, None, False)
    return jnp.argmax(logits, axis=1)


if __name__ == "__main__":
    X_train = read_image("./train-images.idx3-ubyte")
    y_train = read_label("./train-labels.idx1-ubyte")
    X_test = read_image("./t10k-images.idx3-ubyte")
    y_test = read_label("./t10k-labels.idx1-ubyte")

    X_train = X_train.reshape(-1, 1, 28, 28).astype(jnp.float32) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28).astype(jnp.float32) / 255.0

    key = jax.random.PRNGKey(67)
    keys = jax.random.split(key, 6)

    w1 = jax.random.normal(keys[0], (32, 1, 3, 3)) * 0.01
    b1 = jnp.zeros(32)
    w2 = jax.random.normal(keys[1], (64, 32, 3, 3)) * 0.01
    b2 = jnp.zeros(64)
    w3 = jax.random.normal(keys[2], (1600, 10)) * 0.01
    b3 = jnp.zeros(10)

    params = [w1, b1, w2, b2, w3, b3]

    bs = 128
    nepochs = 10_000
    lr = 1e-3
    key = keys[3]
    adam_state = {
        "m": [jnp.zeros_like(p) for p in params],
        "v": [jnp.zeros_like(p) for p in params],
        "t": 0,
    }
    for epoch in range(1, nepochs + 1):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        samp = jax.random.randint(subkey1, (bs,), 0, X_train.shape[0])
        X, y = X_train[samp], y_train[samp]

        params, loss, adam_state = step(params, X, y, subkey2, lr, adam_state)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{nepochs}, Loss: {loss}")

    test_preds = predict(params, X_test)
    acc = jnp.mean(test_preds == y_test)
    print(f"Acc: {acc * 100:.2f}")

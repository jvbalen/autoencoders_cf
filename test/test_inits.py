
import numpy as np
import tensorflow as tf
from scipy.sparse import random

from util import prune, save_weights
from models.skl import SparsePretrainedLR, SparsePretrainedLRFromFile
from models.tf import WAE, prepare_batch


def test_pretrained_lr(tmp_path):
    """Test if SparsePretrainedLR and SparsePretrainedLRFromFile,
    initialized with the same weights, return the same predictions.
    """
    path = str(tmp_path / "weights.npz")
    np.random.seed(1988)

    w = np.random.randn(100, 100)
    b = np.random.rand(100)
    w_sp = prune(w, target_density=0.1)
    save_weights(path, weights=w_sp, biases=b)

    model = SparsePretrainedLR(coefs=w_sp, intercepts=b)
    model_from_file = SparsePretrainedLRFromFile(path)

    x = random(200, 100, density=0.1)
    y = model.predict_logits(x)
    y_from_file = model_from_file.predict_logits(x)

    assert np.allclose(y, y_from_file, rtol=1e-03)


def test_wae_inits(tmp_path):
    """Test if a 1-layer WAE and a logistic regression model,
    initialized with the same weights, return the same predictions.
    """
    np.random.seed(1988)

    w = np.random.randn(100, 100)
    b = np.random.rand(100)
    w_sp = prune(w, target_density=0.1)

    model_np = SparsePretrainedLR(coefs=w_sp, intercepts=b)
    model_tf = WAE(w_inits=[w_sp], b_inits=[b], loss='bce')

    x = random(200, 100, density=0.1)
    y_np = model_np.predict_logits(x)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {model_tf.input_ph: prepare_batch(x)}
        y_tf = sess.run(model_tf.logits, feed_dict=feed_dict)

    assert np.allclose(y_np, y_tf, rtol=1e-03)

# autencoders_cf

Experiments with auto-encoders for collaborative filtering.

This repo is based on [`vae_cf`](https://github.com/dawenl/vae_cf) by Liang, author of [1]. Most of the TensorFlow model code originated there, as well as much of the training and evaluation code.

It also incorporates some refactoring due to [`vae-cf-pytorch`](https://github.com/belepi93/vae-cf-pytorch) by `belepi93`.

## Models

- `models.slim.LinearRecommender`: SLIM [3] in its closed-form variant [2]
- `models.skl.SKLRecommender`: a recommender class for wrapping scikit-learn classifiers
- `models.tf.WAERecommender`: a sparse, full-rank auto-encoder, implemented in TensorFlow

The `WAE` model performs high-dimensional regression, like SLIM [3] and EASE^R [2], but with sparse weights. It uses SGD for optimization, and additionally supports binary and categorical cross-entropy alongside SLIM's squared-error loss.

## Requirements

Python 3.6, and the following packages:
```
numpy
scipy
pandas
scikit-learn
tensorflow==1.15
tensorboard  # if you wish
gin-config
```

## References

[1] *Variational Autoencoders for Collaborative Filtering.* Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman and Tony Jebara, WWW 2018
https://arxiv.org/abs/1802.05814

[2] *Embarrassingly shallow auto-encoders.* Harald Steck, WWW 2019
https://arxiv.org/pdf/1905.03375.pdf

[3] *SLIM: Sparse Linear Methods for Top-N Recommender Systems.* Xia Ning and George Karypis, ICDM 2011
http://glaros.dtc.umn.edu/gkhome/node/774

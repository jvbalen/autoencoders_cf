# autencoders_cf

Experiments with auto-encoders for collaborative filtering.

This repo is based on [`vae_cf`](https://github.com/dawenl/vae_cf) by Liang, author of [1]. Most of the TensorFlow model code originated there, as well as much of the training and evaluation code.

It also incorporates some refactoring due to [`vae-cf-pytorch`](https://github.com/belepi93/vae-cf-pytorch) by `belepi93`.


## Models

Mostly stable implementations:
- `models.linear.LinearRecommender`: simple linear recommender that defaults to SLIM [3] in its closed-form variant [2]. Can be combined with several other non-gradient-based learning algorithms via its `weights_fn` argument. See also `LinearRecommenderFromFile`
- `models.skl.SKLRecommender`: a recommender class for wrapping scikit-learn classifiers 
- `models.als.ALSRecommender`: a simplified implementation of Hu's weighted alternating least squares [4]
- `models.tf.TFRecommender`: a (denoising) auto-encoder recommender implemented in TensorFlow, various models supported

Experimental:
- `models.als.WALSRecommender`: a version of Hu's weighted ALS [4] but with arbitrary dense weights
- `models.als.WSLIMRecommender`: a closed-form version of SLIM with weighting as proposed in Hu's weighted ALS paper [4]
- `models.distill.DistilledRecommender`: a TFRecommender that is trained on the predictions of another model

Baselines:
- `models.base.PopularityRecommender`


## Configuration

This code supports [`gin-config`](https://github.com/google/gin-config/) for configuring experiments.

For example, you can use the following config to run an experiment with the `LinearRecommender`, since the recommender, as well as the `closed_form_slim` were made configurable:
```{python}
experiment.Recommender = @LinearRecommender
LinearRecommender.weights_fn = @closed_form_slim
LinearRecommender.target_density = 0.01
closed_form_slim.l2_reg = 100.0
```

The above is included here as `config/example/slim.gin`. An experiment may look like this:
```{bash}
python run.py --data ~/data/ml-20m --logdir ~/experiments/ml-20m --config confige/examples/slim.gin
```

## Requirements

Python 3.6 and the following packages:
```
numpy
scipy
pandas
scikit-learn
tensorflow==1.15
tensorboard  # optional
gin-config
```

## References

[1] *Variational Autoencoders for Collaborative Filtering.* Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman and Tony Jebara, WWW 2018
https://arxiv.org/abs/1802.05814

[2] *Embarrassingly shallow auto-encoders.* Harald Steck, WWW 2019
https://arxiv.org/pdf/1905.03375.pdf

[3] *SLIM: Sparse Linear Methods for Top-N Recommender Systems.* Xia Ning and George Karypis, ICDM 2011
http://glaros.dtc.umn.edu/gkhome/node/774

[4] *Collaborative filtering for implicit feedback datasets.* Yifan Hu, Yehuda Koren, and Chris Volinsky,  ICDM 2008
https://ieeexplore.ieee.org/document/4781121

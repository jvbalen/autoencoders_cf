# TODO: run on server, as well as the imposter experiment with thr ~ median
python run.py --data ~/data/ml-20m --logdir ~/experiments/tmp --config config/tmp/wals_00.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/tmp --config config/tmp/wals_t-3.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/tmp --config config/tmp/wals_t3.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/tmp --config config/tmp/wals_t-1.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/tmp --config config/tmp/wals_t1.gin

python run.py --data ~/data/ml-20m/ --logdir ~/experiments/autoencoders/ml-20m/ --config config/test_chol_ml20m_200.gin
python run.py --data ~/data/ml-20m/ --logdir ~/experiments/autoencoders/ml-20m/ --config config/test_ease_ml20m_200.gin

python run.py --data ~/data/netflix/ --logdir ~/experiments/autoencoders/netflix/ --config config/test_chol_netflix_200.gin
python run.py --data ~/data/netflix/ --logdir ~/experiments/autoencoders/netflix/ --config config/test_ease_netflix_200.gin

python run.py --data ~/data/msd/ --logdir ~/experiments/autoencoders/msd/ --config config/test_chol_msd_200.gin
python run.py --data ~/data/msd/ --logdir ~/experiments/autoencoders/msd/ --config config/test_ease_msd_200.gin


# dense baseline to save its weights
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_None_None_save.gin

# evaluate from low-rank to high-rank to make first results come in quickly
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_200_None.gin

# from rank 600, evaluate first a sparse model with item_nnz=200, then the dense one
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_600_200.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_600_None.gin

# same for rank 2000 and 6000
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_2000_200.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_2000_None.gin

# from here on, we may want to repeat the experiment but without making XL dense, to compare prediction time
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_6000_200.gin
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_6000_None.gin

# full-rank with item_nnz: the one we have in the paper
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_None_200.gin

# full-rank, dense: same as first model, but useful to confirm save-load works right
python run.py --data ~/data/ml-20m --logdir ~/experiments/www/ml-20m --config config/www_paper/chol_100_None_None.gin

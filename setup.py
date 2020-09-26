from setuptools import setup, find_packages

setup(
    name='autoencoders_cf',
    version='1.0.0',
    url='https://github.com/jvbalen/autoencoders_cf',
    author='Jan Van Balen',
    author_email='jvanbalen@uantwerpen.be',
    description='Autoencoders for Collaborative Filtering',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn',
                      'tensorflow==1.15.4', 'gin-config', 'tqdm'],
    extras_require={
        'test': ['pytest'],
    },
)
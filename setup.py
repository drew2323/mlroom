from setuptools import setup, find_packages

setup(
    name='mlroom',
    version='0.37',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # If the file is in a package, specify the package name and file
        'mlroom': ['config.toml'],
    },
    install_requires=[
        'keras==2.15',  # Specify minimum version requirement
        'tensorflow==2.15',  # Specify minimum version requirement
        'scikit-learn==1.3.2',  # Specify minimum version requirement
        'appdirs',
        'alpaca-py',
        'joblib==1.3.2',
        'keras-tcn'
    ],
)
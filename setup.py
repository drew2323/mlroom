from setuptools import setup, find_packages

setup(
    name='mlroom',
    version='0.2',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # If the file is in a package, specify the package name and file
        'mlroom': ['config.toml'],
    }
)
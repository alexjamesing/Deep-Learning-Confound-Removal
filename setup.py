from setuptools import setup, find_packages

setup(
    name='deepcleaner', 
    version='0.1.0', 
    author='Alex Ing',
    author_email='alexing142@gmail.com', 
    description='A tensorflow/keras package for removing negative effects of confounds/covariates of no interest', 
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown', 
    url='https://github.com/alexjamesing/Deep-Learning-Confound-Removal', # Replace with the URL to your repo
    packages=find_packages(),
    package_data={
        # Include any package containing *.csv files
        'deepcleaner': ['data/*.csv'], 
    },
    include_package_data=True,
    install_requires=[
        'tensorflow', 
        'pandas', 
        'setuptools', # setuptools is required for pkg_resources
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Change the license as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', # Minimum version requirement of the package
)
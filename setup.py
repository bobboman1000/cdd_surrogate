from distutils.core import setup

setup(
    name='cdd-surrogate',
    version='0.0.1',
    packages=['fccd'],
    package_dir={'fccd': 'fccd'},
    url='',
    license='MIT',
    author='Benjamin Jochum',
    author_email='uzebb@student.kit.edu',
    description='fccd python package',
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'jupyter',
        'pytorch-lightning',
        'seaborn',
        'lightgbm'
    ]
)

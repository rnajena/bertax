from setuptools import setup, find_packages

requirements = [
    'tensorflow', 'numpy', 'keras', 'keras-bert'
]

setup(
    name='bertax',
    version=0.1,
    description='DNA sequence taxonomy prediction',
    long_description='',
    license='MIT',
    author='Fleming Kretschmer, Florian Mock, Anton Kriese',
    author_email='fleming.kretschmer@uni-jena.de',
    url='https://github.com/rnajena/bertax',
    packages=find_packages(),
    package_data={'bertax': ['resources/big_trainingset_all_fix_classes_selection.h5']},
    entry_points={
        'console_scripts': [
            'bertax=bertax.bertax:main',
            'bertax-visualize=bertax.visualize:main'
        ]
    },
    install_requires=requirements,
    python_requires=">=3.7"
)

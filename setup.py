from setuptools import setup,find_packages

setup(
    name = "NLP tools lib",
    version = 0.1,
    scripts=find_packages(),

    install_requires=[
        'deepmultilingualpunctuation'
        'spacy',
        'tqdm',
        'pandas',
        're',

    ]
)
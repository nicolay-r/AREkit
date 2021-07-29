from setuptools import (
    setup,
    find_packages,
)


def get_requirements(filenames):
    r_total = []
    for filename in filenames:
        with open(filename) as f:
            r_local = f.read().splitlines()
            r_total.extend(r_local)
    return r_total

setup(
    name='arekit',
    version='0.21.0',
    description='Library devoted to Document level Attitude and Relation Extraction '
                'for text objects with entity-linking (EL) API support',
    url='https://github.com/nicolay-r/AREkit',
    author='Nicolay Rusnachenko',
    author_email='???',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='natural language processing, relation extraction, sentiment analysis',
    packages=find_packages(),
    install_requires=get_requirements([
        'dependencies.txt',
        'arekit/processing/dependencies.txt',
        'arekit/contrib/networks/dependencies.txt']),
)
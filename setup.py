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
    version='0.25.1',
    python_requires=">=3.6",
    description='Document level Attitude and Relation Extraction toolkit (AREkit)'
                ' for sampling and prompting mass-media news into datasets for ML-model training',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nicolay-r/AREkit',
    author='Nicolay Rusnachenko',
    author_email='rusnicolay@gmail.com',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='natural language processing, relation extraction, sentiment analysis',
    packages=find_packages(),
    install_requires=get_requirements([
        'dependencies.txt']),
    data_files=["logo.png"],
)

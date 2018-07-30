import os
import sys
import re
from glob import glob
from setuptools import setup, find_packages, Extension


__pkg_name__ = 'scrappy'
__author__ = 'cwright'
__description__ = 'Python bindings to scrappie basecaller'

# Use readme as long description and say its github-flavour markdown
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), 'rb') as f:
    __long_description__ = f.read().decode('utf-8')
__long_description_content_type__ = 'text/markdown'


__path__ = os.path.dirname(__file__)
__pkg_path__ = os.path.join(os.path.join(__path__, __pkg_name__))

# Get the version number from __init__.py
verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


# Get requirements from file, we prefer to have
#   preinstalled these with pip to make use of wheels.
dir_path = os.path.dirname(__file__)
install_requires = []
with open(os.path.join(dir_path, 'requirements.txt')) as fh:
    reqs = (
        r.split('#')[0].strip()
        for r in fh.read().splitlines() if not r.startswith('#')
    )
    for req in reqs:
        if req == '':
            continue
        if req.startswith('git+https'):
            req = req.split('/')[-1].split('@')[0]
        install_requires.append(req)

extra_requires = {}
additional_tests_requires = ['nose>=1.0']
extensions = []

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    'Natural Language :: English',
    'Operating System :: Unix',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics'
]


if 'MANYLINUX' in os.environ:
    # when building wheel we need to include this
    data_files = [('', ['OPENBLAS_LICENSE'])]
else:
    data_files = []

setup(
    name='scrappie', # scrappy is taken on pypi
    version=__version__,
    url='https://github.com/nanoporetech/scrappie',
    author=__author__,
    author_email='{}@nanoporetech.com'.format(__author__),
    classifiers=classifiers,
    description=__description__,
    long_description=__long_description__,
    long_description_content_type=__long_description_content_type__,
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    license='MPL 2.0',

    dependency_links=[],
    ext_modules=extensions,
    install_requires=install_requires,
    tests_require=install_requires + additional_tests_requires,
    extras_require=extra_requires,
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["build.py:ffibuilder"],
    # don't include any testing subpackages in dist
    packages=find_packages(exclude=['*.test', '*.test.*', 'test.*', 'test']),
    package_data={},
    zip_safe=False,
    data_files=data_files,
    entry_points={
        'console_scripts': [
            '{} = {}:_basecall'.format(__pkg_name__, __pkg_name__)
        ]
    },
    scripts=[
    ],
    test_suite='nose.collector',
)

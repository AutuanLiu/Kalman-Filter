"""setup

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/01/08
"""

import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages, setup

version = '0.5.4'

if sys.argv[1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git pull origin master',
        'git tag v{:s}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist',
        'twine upload dist/imgviz-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def get_install_requires():
    install_requires = []
    with open('requirements.txt') as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


with open('description.md') as f:
    long_description = f.read()

setup(
    name='kalman-estimation',
    version=version,
    packages=find_packages(),
    install_requires=get_install_requires(),
    description='Kalman filter based coefficient estimation toolbox.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    python_requires='>=3.5',
    author='Autuan Liu',
    author_email='autuanliu@163.com',
    url='https://github.com/AutuanLiu/Kalman-Filter',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)

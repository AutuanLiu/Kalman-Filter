import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages, setup

import github2pypi

version = '0.1.0'

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
    long_description = github2pypi.replace_url(slug='AutuanLiu/Kalman-Filter', content=f.read())

setup(
    name='k4est',
    version=version,
    packages=find_packages(),
    install_requires=get_install_requires(),
    description='Kalman filter based coefficient estimation toolbox.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={'data': ['*']},
    include_package_data=True,
    author='Autuan Liu',
    author_email='autuanliu@g163.com',
    url='https://github.com/AutuanLiu/Kalman-Filter',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English, Chinese',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)

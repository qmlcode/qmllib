python=python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip

all: env

env:
	${mamba} env create -f ./environment_dev.yaml -p ./env --quiet
	${pip} install -e .

setup:
	pre-commit install

format:
	pre-commit run --all-files

test:
	${python} -m pytest -rs tests

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

python=./env/bin/python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip
pytest=pytest
j=1

.PHONY: build

all: env

## Setup

env:
	${mamba} env create -f ./environment_dev.yaml -p ./env --quiet
	${python} -m pre_commit install
	${python} -m pip install -e .

./.git/hooks/pre-commit:
	${python} -m pre_commit install

## Development

format:
	${python} -m pre_commit run --all-files

test:
	${python} -m pytest -rs ./tests

types:
	${python} -m monkeytype run $$(which ${pytest}) ./tests
	${python} -m monkeytype list-modules | grep ${pkg} | parallel -j${j} "${python} -m monkeytype apply {}"

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

compile:
	${python} _compile.py

build:
	${python} -m build --sdist --skip-dependency-check  .

upload:
	${python} -m twine upload ./dist/*.tar.gz

## Clean

clean:
	find ./src/ -type f \
		-name "*.so" \
		-name "*.pyc" \
		-name ".pyo" \
		-delete
	rm -rf ./src/*.egg-info/
	rm -rf *.whl
	rm -rf ./build/ ./__pycache__/
	rm -rf ./dist/

clean-env:
	rm -rf ./env/
	rm ./.git/hooks/pre-commit

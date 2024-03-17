python=./env/bin/python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip
pytest=pytest
j=1

.PHONY: build

all: env setup

env:
	${mamba} env create -f ./environment_dev.yaml -p ./env --quiet
	${pip} install -e .

setup: ./.git/hooks/pre-commit

./.git/hooks/pre-commit:
	${python} -m pre_commit install

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
	@#${python} -m build .
	# ${python} -m pip wheel --no-deps -v .
	${python} -m pip wheel -v .
	ls *.whl

clean:
	find ./src/ -type f \
		-name "*.so" \
		-name "*.pyc" \
		-name ".pyo" \
		-delete
	rm -rf *.whl
	rm -fr ./build/ ./__pycache__/

clean-env:
	rm -rf ./env/
	rm ./.git/hooks/pre-commit

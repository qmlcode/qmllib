python=./env/bin/python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip
j=1

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
	${python} -m pytest -rs \
	./tests/test_kernels.py \
	./tests/test_solvers.py \
	./tests/test_distance.py \
	./tests/test_slatm.py

types:
	${python} -m monkeytype run $(which pytest) ./tests/
	${python} -m monkeytype list-modules | grep ${pkg} | parallel -j${j} "${python} -m monkeytype apply {}"

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

compile:
	${python} _compile.py

build: compile
	${python} -m build .

clean:
	find ./src/ | grep -E "\(/__pycache__$$|\.pyc$$|\.pyo$$\|\.so$$)" | xargs rm -rf

clean-env:
	rm -rf ./env/
	rm ./.git/hooks/pre-commit

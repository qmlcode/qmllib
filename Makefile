python=./env/bin/python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip

all: env setup

env:
	${mamba} env create -f ./environment_dev.yaml -p ./env --quiet
	${pip} install -e .

setup: ./.git/hooks/pre-commit

./.git/hooks/pre-commit:
	${python} -m pre_commit install

format:
	${python} pre_commit run --all-files

test:
	${python} -m pytest -rs ./tests/test_kernels.py

types:
	# ${python} -m monkeytype run $(which pytest) ./tests/test_solvers.py
	${python} -m monkeytype list-modules | grep ${pkg} | parallel -j1 "${python} -m monkeytype apply {}"

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

compile:
	${python} _compile.py

build: compile
	${python} -m build .

clean:
	rm -rf ./env/
	find ./src/ | grep -E "\(/__pycache__$$|\.pyc$$|\.pyo$$\)" | xargs rm -rf
	rm ./.git/hooks/pre-commit

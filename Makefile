python=./env/bin/python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip
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
	${python} -m pytest -rs \
		tests/test_distance.py \
		tests/test_kernels.py \
		tests/test_representations.py \
		tests/test_slatm.py \
		tests/test_solvers.py \
		tests/test_fchl_acsf.py \
		tests/test_fchl_acsf_energy.py \
		tests/test_fchl_acsf_forces.py \
		# tests/test_fchl_electric_field.py \
		# tests/test_fchl_force.py \
		# tests/test_fchl_scalar.py
	# integration tests/test_energy_krr_atomic_cmat.py \
	# integration tests/test_energy_krr_bob.py \
	# integration tests/test_energy_krr_cmat.py \
	# tests/test_kernel_derivatives.py \
	# tests/test_arad.py \

types:
	${python} -m monkeytype run $(which pytest) ./tests/
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

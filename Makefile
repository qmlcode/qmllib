all: .venv

install:
	uv pip install -e .[test,dev] --verbose

install-native:
	CMAKE_ARGS="-DQMLLIB_USE_NATIVE=ON" uv pip install -e .[test,dev] --verbose

install-dev:
	pip install -e .[test,dev] --verbose
	pre-commit install

test:
	uv run pytest -m "not integration" tests/ -v -s

test-all:
	uv run pytest tests/ -v -s

.venv:
	uv venv --python 3.14
	# These are sometimes missed in GitHub CI builds
	uv pip install scikit-build-core pybind11

check: format types

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix --verbose src/ tests/

types:
	uv run ty check src/ --exclude tests/

stubs:
	mkdir -p stubs_temp
	stubgen -p qmllib._fdistance -o stubs_temp
	stubgen -p qmllib._fgradient_kernels -o stubs_temp
	stubgen -p qmllib._fkernels -o stubs_temp
	stubgen -p qmllib._facsf -o stubs_temp
	stubgen -p qmllib._representations -o stubs_temp
	stubgen -p qmllib._fslatm -o stubs_temp
	stubgen -p qmllib._solvers -o stubs_temp
	stubgen -p qmllib._utils -o stubs_temp
	stubgen -p qmllib.representations.fchl.ffchl_module -o stubs_temp
	mv stubs_temp/qmllib/*.pyi src/qmllib/
	mv stubs_temp/qmllib/representations/fchl/ffchl_module/*.pyi src/qmllib/representations/fchl/
	# rm -rf stubs_temp
	uv run ruff format src/qmllib/**/*.pyi

clean:
	find ./src/ -type f \
		-name "*.so" \
		-name "*.pyc" \
		-name ".pyo" \
		-name ".mod" \
		-delete
	rm -rf ./src/*.egg-info/
	rm -rf *.whl
	rm -rf ./build/ ./__pycache__/
	rm -rf ./dist/

clean-env:
	rm -rf ./.venv/
	rm -rf ./.git/hooks/pre-commit

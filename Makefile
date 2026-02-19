.PHONY: install install-dev test test-all test-integration check format typing stubs clean help

install:
	uv pip install -e .[test,dev] --verbose

install-native:
	CMAKE_ARGS="-DQMLLIB_USE_NATIVE=ON" uv pip install -e .[test,dev] --verbose

install-dev:
	pip install -e .[test,dev] --verbose
	pre-commit install

# Run fast unit tests only (exclude integration tests)
test:
	uv run pytest -m "not integration" tests/ -v -s

# Run all tests including integration tests
test-all:
	uv run pytest tests/ -v -s

env_uv:
	uv venv --python 3.14
	# These are sometimes missed in GitHub CI builds
	uv pip install scikit-build-core pybind11

check: format typing

format:
	ruff format src/ tests/
	ruff check --fix --verbose src/ tests/

types:
	ty check src/ --exclude tests/

stubs:
	@echo "Generating type stubs for Fortran/pybind11 modules..."
	@mkdir -p stubs_temp
	stubgen -p qmllib._fdistance -o stubs_temp
	stubgen -p qmllib._fgradient_kernels -o stubs_temp
	stubgen -p qmllib._fkernels -o stubs_temp
	stubgen -p qmllib._facsf -o stubs_temp
	stubgen -p qmllib._representations -o stubs_temp
	stubgen -p qmllib._fslatm -o stubs_temp
	stubgen -p qmllib._solvers -o stubs_temp
	stubgen -p qmllib._utils -o stubs_temp
	stubgen -p qmllib.representations.fchl.ffchl_module -o stubs_temp
	@echo "Moving stubs to src/qmllib/..."
	@mv stubs_temp/qmllib/*.pyi src/qmllib/
	@mv stubs_temp/qmllib/representations/fchl/ffchl_module/*.pyi src/qmllib/representations/fchl/ || true
	@rm -rf stubs_temp
	@echo "Formatting stub files with ruff..."
	ruff format src/qmllib/**/*.pyi
	@echo "Stubs generated and formatted successfully!"

clean:                             
	find ./src/ -type f \          
		-name "*.so" \             
		-name "*.pyc" \            
		-name ".pyo" \             
		-name ".mod" \             
		-delete                    
	rf ./src/*.egg-info/       
	rm -rf *.whl                   
	rm -rf ./build/ ./__pycache__/ 
	rm -rf ./dist/                 
                                   
clean-env:                         
	rm -rf ./env/                  
	rm ./.git/hooks/pre-commit     


environment:
	conda env create -f environments/environment-dev.yaml

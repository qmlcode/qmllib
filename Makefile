.PHONY: install install-dev test test-all test-integration check format typing stubs clean help

install:
	pip install -e .[test] --verbose

install-dev:
	pip install -e .[test,dev] --verbose
	pre-commit install

# Run fast unit tests only (exclude integration tests)
test:
	pytest -m "not integration"

# Run all tests including integration tests
test-all:
	pytest

# Run only integration tests
test-integration:
	pytest -m integration

check: format typing

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

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

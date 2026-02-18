.PHONY: install install-dev test check format typing clean help

install:
	pip install -e .[test] --verbose

install-dev:
	pip install -e .[test,dev] --verbose
	pre-commit install

test:
	pytest

check: format typing

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

types:
	mypy src/ tests/

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

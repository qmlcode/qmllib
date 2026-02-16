install:
	pip install -e .[test] --verbose

test:
	pytest

environment:
	conda env create -f environments/environment-dev.yaml

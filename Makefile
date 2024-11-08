python=./env/bin/python
mamba=mamba
pkg=qmllib
pip=./env/bin/pip
pytest=pytest
j=1

version_file=src/qmllib/version.py

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
	${python} -m monkeytype list-modules | grep ${pkg} | parallel -j${j} "${python} -m monkeytype apply {} > /dev/null && echo {}"

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

compile:
	${python} _compile.py

build:
	${python} -m build --sdist --skip-dependency-check  .

upload:
	${python} -m twine upload ./dist/*.tar.gz

## Version

VERSION=$(shell cat ${version_file} | egrep -o "([0-9]{1,}\.)+[0-9]{1,}")
VERSION_PATCH=$(shell echo ${VERSION} | cut -d'.' -f3)
VERSION_MINOR=$(shell echo ${VERSION} | cut -d'.' -f2)
VERSION_MAJOR=$(shell echo ${VERSION} | cut -d'.' -f1)
GIT_COMMIT=$(shell git rev-parse --short HEAD)

version:
	echo ${VERSION}

bump-version-auto:
	test $(git diff HEAD^ HEAD tests | grep -q "+def") && make bump-version-minor || make bump-version-patch

bump-version-dev:
	test ! -z "${VERSION}"
	test ! -z "${GIT_COMMIT}"
	exit 1 # Not Implemented

bump-version-patch:
	test ! -z "${VERSION_PATCH}"
	echo "__version__ = \"${VERSION_MAJOR}.${VERSION_MINOR}.$(shell awk 'BEGIN{print ${VERSION_PATCH}+1}')\"" > ${version_file}

bump-version-minor:
	test ! -z "${VERSION_MINOR}"
	echo "__version__ = \"${VERSION_MAJOR}.$(shell awk 'BEGIN{print ${VERSION_MINOR}+1}').0\"" > ${version_file}

bump-version-major:
	test ! -z "${VERSION_MAJOR}"
	echo "__version__ = \"$(shell awk 'BEGIN{print ${VERSION_MAJOR}+1}').0.0\"" > ${version_file}

commit-version-tag:
	# git tag --list | grep -qix "${VERSION}"
	git commit -m "Release ${VERSION}" --no-verify ${version_file}
	git tag 'v${VERSION}'

gh-release:
	gh release create "v${VERSION}" \
	--repo="$${GITHUB_REPOSITORY}" \
	--title="$${GITHUB_REPOSITORY#*/} ${VERSION}" \
	--generate-notes

gh-has-src-changed:
	git diff HEAD^ HEAD src | grep -q "+"

gh-cancel:
	gh run cancel $${GH_RUN_ID}
	gh run watch $${GH_RUN_ID}

## Clean

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
	rm -rf ./env/
	rm ./.git/hooks/pre-commit

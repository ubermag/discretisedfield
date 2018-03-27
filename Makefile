PROJECT=discretisedfield
IPYNBPATH=docs/ipynb/*.ipynb
CODECOVTOKEN=a96f2545-67ea-442e-a1b6-fdebc595cf52
PYTHON?=python3

# run all tests that can be found in source directories
test:
	$(PYTHON) -m pytest

# run tests that users can activate via 'discretisedfield.test()'
test-test:
	$(PYTHON) -c "import sys; import discretisedfield as d; sys.exit(d.test())"

# like 'test' but gather coverage
test-coverage:
	$(PYTHON) -m pytest --cov=$(PROJECT) --cov-config .coveragerc

# NBVAL to check notebooks for regression
test-ipynb:
	$(PYTHON) -m pytest --nbval $(IPYNBPATH)

# run doctests (?)
test-docs:
	$(PYTHON) -m pytest --doctest-modules --ignore=$(PROJECT)/tests $(PROJECT)

# do all ('test' is the same set of tests as 'test-coverage')
test-all: test-test test-coverage test-ipynb test-docs

upload-coverage: SHELL:=/bin/bash
upload-coverage:
	bash <(curl -s https://codecov.io/bash) -t $(CODECOVTOKEN)

travis-build: SHELL:=/bin/bash
travis-build:
	ci_env=`bash <(curl -s https://codecov.io/env)`
	docker build -t dockertestimage .
	docker run -e ci_env -ti -d --name testcontainer dockertestimage
	docker exec testcontainer make test-test
	docker exec testcontainer make upload-coverage
	docker stop testcontainer
	docker rm testcontainer

test-docker:
	make travis-build

build-dists:
	rm -rf dist/
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel

release: build-dists
	twine upload dist/*

PROJECT=discretisedfield
IPYNBPATH=docs/ipynb/*.ipynb
CODECOVTOKEN=ec842dd4-be45-4f5d-acfb-66857fa13de0
PYTHON?=python3

test:
	$(PYTHON) -m pytest

test-test:
	$(PYTHON) -c "import sys; import $(PROJECT); sys.exit($(PROJECT).test())"

test-coverage:
	$(PYTHON) -m pytest --cov=$(PROJECT) --cov-config .coveragerc

test-docs:
	$(PYTHON) -m pytest --doctest-modules --ignore=$(PROJECT)/tests $(PROJECT)

test-ipynb:
	$(PYTHON) -m pytest --nbval $(IPYNBPATH)

test-all: test-test test-coverage test-docs

upload-coverage: SHELL:=/bin/bash
upload-coverage:
	bash <(curl -s https://codecov.io/bash) -t $(CODECOVTOKEN)

travis-build: SHELL:=/bin/bash
travis-build:
	ci_env=`bash <(curl -s https://codecov.io/env)`
	docker build -f docker/Dockerfile -t dockertestimage .
	docker run -e ci_env -ti -d --name testcontainer dockertestimage
	docker exec testcontainer make test-all
	docker exec testcontainer make upload-coverage
	docker stop testcontainer
	docker rm testcontainer

test-docker:
	docker build -f docker/Dockerfile -t dockertestimage .
	docker run -ti -d --name testcontainer dockertestimage
	docker exec testcontainer find . -name '*.pyc' -delete
	docker exec testcontainer make test-all
	docker stop testcontainer
	docker rm testcontainer

build-dists:
	rm -rf dist/
	$(PYTHON) setup.py sdist bdist_wheel

release: build-dists
	twine upload dist/*

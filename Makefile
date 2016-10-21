PROJECT=discretisedfield
IPYNBPATH=docs/ipynb/*.ipynb
CODECOVTOKEN=a96f2545-67ea-442e-a1b6-fdebc595cf52

test: test-coverage test-ipynb

test-all:
	python3 -m pytest

test-ipynb:
	python3 -m pytest --nbval $(IPYNBPATH)

test-coverage:
	python3 -m pytest --cov=$(PROJECT) --cov-config .coveragerc

upload-coverage: SHELL:=/bin/bash
upload-coverage:
	bash <(curl -s https://codecov.io/bash) -t $(CODECOVTOKEN)

travis-build: test-coverage upload-coverage

test-docker:
	docker build -t dockertestimage .
	docker run --privileged -ti -d --name testcontainer dockertestimage
	docker exec testcontainer python3 -m pytest
	docker exec testcontainer python3 -m pytest --nbval $(IPYNBPATH)
	docker stop testcontainer
	docker rm testcontainer

pypitest-upload:
	python3 setup.py register -r pypitest
	python3 setup.py sdist upload -r pypitest

pypi-upload: pypitest-upload
	python3 setup.py sdist upload -r pypi

test: test-code test-ipynb

test-code:
	py.test --cov=discretisedfield --cov-config .coveragerc

test-ipynb:
	py.test --nbval docs/ipynb/*.ipynb


testd:
	# run TESTs in Docker container (TESTD). The commands
	# below are copied from .travis.yml (excluding coverage tool update)
	# This is a convenience target to run the Travis tests (inside container) locally.
	docker build -t dockertestimage .
	# run tests in docker
	docker run -e ci_env -ti dockertestimage /bin/bash -c "cd discretisedfield && python3 -m pytest --cov=discretisedfield --cov-config .coveragerc && python3 -m pytest --nbval docs/ipynb/*.ipynb"

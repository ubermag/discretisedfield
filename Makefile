test: test-code test-ipynb

test-code:
	py.test --cov=discretisedfield --cov-config .coveragerc

test-ipynb:
	py.test --nbval docs/ipynb/*.ipynb

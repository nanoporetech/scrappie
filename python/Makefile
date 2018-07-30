.PHONY: install clean sdist test_sdist

venv: venv/bin/activate
IN_VENV=. ./venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv --python=python3
	${IN_VENV} && pip install pip --upgrade

install: clean venv
	${IN_VENV} && pip install -r requirements.txt && python setup.py install

sdist: clean venv
	cp -r ../src src
	python setup.py sdist

test_sdist: sdist
	mkdir -p test_sdist
	cd test_sdist && tar -xzvf ../dist/scrappie-*.tar.gz
	${IN_VENV} && cd test_sdist/scrappie-* && python setup.py install && python -c "from scrappy import *; import numpy as np; print(basecall_raw(np.random.normal(10,4,1000)))"

wheels:
	cd .. && docker run -v `pwd`:/io ontresearch/manylinux1_x86_64_openblas /io/python/build-wheels.sh

clean:
	rm -rf lib venv build dist libscrappy.* *.egg-info src test_sdist wheelhouse

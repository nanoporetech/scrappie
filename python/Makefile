.PHONY: install clean

venv: venv/bin/activate
IN_VENV=. ./venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv --python=python3
	${IN_VENV} && pip install pip --upgrade

install: clean venv
	${IN_VENV} && pip install -r requirements.txt && python setup.py install

clean:
	rm -rf lib venv build dist *.egg-info

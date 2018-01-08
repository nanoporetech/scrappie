.PHONY: install clean

venv: venv/bin/activate
IN_VENV=. ./venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv --python=python3
	${IN_VENV} && pip install pip --upgrade

install: clean venv lib/libscrappie.a 
	${IN_VENV} && pip install -r requirements.txt && python setup.py install

lib/libscrappie.a:
	mkdir -p lib 
	cd lib && cmake ../.. && make scrappie_static

clean:
	rm -rf lib venv build dist *.egg-info

.PHONY: install-dev clean format check install uninstall test diff-test


install-dev:
	python3 -m pip install \
		pytest==7.1.2 \
		matplotlib \
		ipython \
		git+https://github.com/sotetsuk/MinAtar.git \
		jax[cpu] \
		brax \
		argdcls \
		tqdm \
		shanten_tools \

install-fmt:
	python3 -m pip install \
		black==22.6.0 \
		blackdoc==0.3.6 \
		isort==5.10.1 \
		flake8==5.0.4 \
		mypy==0.971

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

format:
	black pgx --exclude=pgx/flax
	blackdoc pgx --exclude=pgx/flax
	isort pgx --skip-glob=pgx/flax

check:
	black pgx --check --diff --exclude=pgx/flax
	blackdoc pgx --check --exclude=pgx/flax
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 pgx --exclude=pgx/flax
	mypy --config pyproject.toml pgx --exclude=pgx/flax/* 
	isort pgx --check --diff --skip-glob=pgx/flax

install:
	python3 -m pip install --upgrade pip setuptools
	python3 setup.py install

uninstall:
	python3 -m pip uninstall pgx -y

test:
	python3 -m pytest --doctest-modules --verbose pgx tests/test_*.py --ignore=pgx/experimental

test-modified:
	./test_modified.sh

.PHONY: install-dev clean format check install uninstall test diff-test


install-dev:
	python3 -m pip install \
		pytest==7.1.2 \
		matplotlib \
		ipython \
		git+https://github.com/sotetsuk/MinAtar.git \
		jax[cpu]

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
	black pgx --exclude=pgx/_flax
	blackdoc pgx --exclude=pgx/_flax
	isort pgx --skip-glob=pgx/_flax

check:
	black pgx --check --diff --exclude=pgx/_flax
	blackdoc pgx --check --exclude=pgx/_flax
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 pgx --exclude=pgx/_flax
	mypy --config pyproject.toml pgx --exclude=pgx/_flax/* 
	isort pgx --check --diff --skip-glob=pgx/_flax

install:
	python3 -m pip install --upgrade pip setuptools
	python3 -m pip install .

uninstall:
	python3 -m pip uninstall pgx -y

test:
	python3 -m pytest --doctest-modules --verbose pgx tests/test_*.py --ignore=pgx/experimental

test-modified:
	./test_modified.sh

.PHONY: install-dev clean format check install uninstall test diff-test


install-dev:
	python3 -m pip install -U \
		pytest \
		matplotlib \
		ipython \
		jax[cpu] \
		dm-haiku \
		pytest-cov \
		pgx-minatar

install-fmt:
	python3 -m pip install -U \
		black \
		blackdoc \
		isort \
		flake8 \
		mypy

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

format:
	black pgx
	blackdoc pgx
	isort pgx

check:
	black pgx --check --diff
	blackdoc pgx --check
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 pgx
	mypy --config pyproject.toml pgx
	isort pgx --check --diff

install:
	python3 -m pip install --upgrade pip setuptools
	python3 -m pip install .

uninstall:
	python3 -m pip uninstall pgx -y

test:
	python3 -m pytest --doctest-modules --verbose pgx tests/test_*.py --ignore=pgx/experimental

test-modified:
	./test_modified.sh

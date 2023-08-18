.PHONY: install-dev clean format check install uninstall test diff-test


install-dev:
	python3 -m pip install \
		pytest==7.1.2 \
		matplotlib \
		ipython \
		jax[cpu] \
		dm-haiku \
		pytest-cov

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
	python3 -m pytest --doctest-modules --verbose pgx tests/test_*.py --ignore=pgx/experimental --cov=./ --cov-report=term-missing --cov-report=xml

test-modified:
	./test_modified.sh

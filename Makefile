.PHONY: install-dev clean format check install uninstall test


install-dev:
	python3 -m pip install -r requirements-dev.txt

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
	python3 setup.py install

uninstall:
	python3 -m pip uninstall pgx -y

test:
	python3 -m pytest --doctest-modules --verbose pgx tests

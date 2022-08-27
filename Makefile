install:
	poetry install --no-interaction

	# Configure pre-comit
	poetry run pre-commit install && poetry run pre-commit install --hook-type pre-push

build:
	poetry build

lint:
	poetry run pre-commit run --all-files

test:
	poetry run pytest tests -n auto

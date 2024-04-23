black:
	poetry run black . --check

ruff:
	poetry run ruff check edit_gpt tests

run:
	poetry run python -m edit_gpt

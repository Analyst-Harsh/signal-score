.PHONY: install lint format format-check typecheck test check

install:
	uv sync --group dev --extra ml
	uv run lefthook install

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest -n auto --cov=signalscore --cov-report=term-missing

check: lint format-check typecheck test

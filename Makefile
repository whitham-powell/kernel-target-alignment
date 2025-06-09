.PHONY: dev env test

env:			## (Re)sync venv with dev extras
	uv pip install -e '.[dev]'

add=%  # make add name=pandas
add:
	uv add -g dev $(name)

test:           ## run pytest under uv
	uv run pytest -q

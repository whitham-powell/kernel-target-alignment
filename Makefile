.PHONY: dev env test plots

env:			## (Re)sync venv with dev extras
	uv pip install -e '.[dev]'

add=%  # make add name=pandas
add:
	uv add -g dev $(name)

test:           ## run pytest under uv
	uv run pytest -q

plots: env         ## Extract plots from executed notebooks (requires dev env)
	@mkdir -p extracted_figures
	@for nb in notebooks/**/*.ipynb; do \
		echo "Executing $$nb..."; \
		uv run jupyter nbconvert "$$nb" \
			--to html \
			--execute \
			--ExecutePreprocessor.timeout=300 \
			--ExtractOutputPreprocessor.enabled=True; \
	done
	@find . -type d -name "*_files" | while read dir; do \
		cp $$dir/* extracted_figures/; \
		rm -r $$dir; \
	done
	@find notebooks -name '*.html' -delete
	@echo "Plots extracted to extracted_figures/"

md: env
	@for nb in notebooks/**/*.ipynb; do \
		echo "Converting $$nb to Markdown..."; \
		uv run jupyter nbconvert "$$nb" \
			--to markdown \
			--execute \
			--ExecutePreprocessor.timeout=300 \
			--output-dir=notebooks/markdown_exports; \
	done

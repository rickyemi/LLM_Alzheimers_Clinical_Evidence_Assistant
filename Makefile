.PHONY: install index test eval agent notebooks figures lint api docker clean
install:   ; pip install -e ".[dev,notebooks]"
index:     ; python scripts/build_index.py
test:      ; python -m pytest -q
eval:      ; python scripts/run_eval.py           # exits 1 if a release gate fails
agent:     ; python scripts/run_agent.py
notebooks: ; python scripts/run_notebooks.py      # executes and saves outputs
figures:   ; python scripts/make_figures.py
lint:      ; ruff check src tests scripts
api:       ; uvicorn ad_rag.api:app --reload --port 8000
docker:    ; docker build -t ad-rag:0.1.0 .
clean:     ; rm -rf artifacts .pytest_cache .ruff_cache

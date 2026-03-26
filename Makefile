.PHONY: run dev test

run:
	uv run uvicorn podmemory.main:app --host 0.0.0.0 --port 8000

dev:
	uv run uvicorn podmemory.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -v

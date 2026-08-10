.PHONY: install playground serve

install:
	uv sync

playground:
	uv run adk web expense_agent --port 8081

serve:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8080

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	LITELLM_LOG="ERROR" uv run agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml

eval: generate-traces grade
	LITELLM_LOG="ERROR" uv run agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml

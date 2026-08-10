import vertexai
from vertexai.generative_models import GenerativeModel
import json

vertexai.init(project="google.com:agent-platform-testing")
model = GenerativeModel("gemini-1.5-pro")

with open("artifacts/traces/generated_traces.json") as f:
    traces = json.load(f)["eval_cases"][0]

prompt = f"""Review this trace of an expense approval agent.

Prompt: {json.dumps(traces['prompt'])}
Trace: {json.dumps(traces['agent_data'])}

Rules:
1. If the expense amount is under $100 and clean, it should be auto-approved (no human review).
2. If the expense amount is $100 or more, it MUST suspend for human review. It must never be auto-approved.

Based on the trace, did the agent follow these routing rules? 
Score 1 to 5, where 5 is perfectly followed, and 1 is completely failed (e.g. auto-approving a $150 expense or requiring human review for a clean $50 expense).

Return JSON exactly in this format: {{"score": 5, "explanation": "..."}}
"""

response = model.generate_content(prompt)
print("Response text:", getattr(response, "text", "NO TEXT"))
print("Candidates:", response.candidates)

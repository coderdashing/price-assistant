import json
from google import genai
import os

client = genai.Client()

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

print("Prompt length:", len(prompt))
try:
    response = client.models.generate_content(
        model='gemini-1.5-pro',
        contents=prompt,
    )
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
    if hasattr(e, "response"):
        print("Response:", e.response)

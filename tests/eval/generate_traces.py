import asyncio
import json
import os
from expense_agent.agent import app
from google.adk.events.request_input import RequestInput
from google.adk.runners import InMemoryRunner
from google.genai import types

runner = InMemoryRunner(app=app)

async def generate():
    with open("tests/eval/datasets/basic-dataset.json") as f:
        dataset = json.load(f)
        
    out_cases = []
    
    for case in dataset["eval_cases"]:
        case_id = case["eval_case_id"]
        text = case["prompt"]["parts"][0]["text"]
        print(f"Running {case_id}...")
        
        trace_log = []
        trace_log.append(f"USER: {text}")
        
        should_approve = "clean" in case_id
        session_id = f"eval_session_{case_id}"
        session = await runner.session_service.create_session(
            app_name=app.name, user_id="eval_user", session_id=session_id
        )
        
        suspended = False
        new_msg = types.Content(role="user", parts=[types.Part.from_text(text=text)])
        
        async for event in runner.run_async(user_id="eval_user", session_id=session_id, new_message=new_msg):
            # Check for adk_request_input function call (which is how ADK 2.0 signals suspension to the runner)
            if getattr(event, "content", None) and hasattr(event.content, "parts") and len(event.content.parts) > 0:
                part = event.content.parts[0]
                if getattr(part, "function_call", None) and getattr(part.function_call, "name", "") == "adk_request_input":
                    msg = part.function_call.args.get("message", "")
                    trace_log.append(f"AGENT SUSPENDED (RequestInput): {msg}")
                    suspended = True
                    break
                
                # Standard text message
                part_text = getattr(part, "text", None)
                if part_text is not None:
                    trace_log.append(f"AGENT MESSAGE: {part_text}")
                else:
                    trace_log.append(f"AGENT CONTENT: {str(part)}")
            elif getattr(event, "output", None):
                trace_log.append(f"NODE OUTPUT ({getattr(event, 'route', 'none')}): {event.output}")
        
        if suspended:
            answer = "yes" if should_approve else "no"
            trace_log.append(f"USER (simulated human): {answer}")
            ans_msg = types.Content(role="user", parts=[types.Part.from_text(text=answer)])
            # Simulate the user hitting the resume endpoint
            # Actually ADK HITL resumes by passing a dict with the interrupt_id
            ans_data = {"approve_expense": answer}
            # For run_async, we just pass it as the new_message but ADK HITL often uses resume_inputs inside the event or the Content
            async for event in runner.run_async(user_id="eval_user", session_id=session_id, new_message=ans_msg):
                if getattr(event, "content", None):
                    content = event.content
                    if hasattr(content, "parts") and len(content.parts) > 0:
                        part_text = getattr(content.parts[0], "text", str(content.parts[0]))
                        trace_log.append(f"AGENT MESSAGE: {part_text}")
                    else:
                        trace_log.append(f"AGENT CONTENT: {content}")
                elif getattr(event, "output", None):
                    trace_log.append(f"NODE OUTPUT ({getattr(event, 'route', 'none')}): {event.output}")

        events = []
        for log_line in trace_log:
            events.append({"author": "system", "content": {"parts": [{"text": log_line}]}})

        out_cases.append({
            "eval_case_id": case_id,
            "prompt": case["prompt"],
            "responses": [
                {
                    "response": {
                        "role": "model",
                        "parts": [{"text": "Evaluation completed"}]
                    }
                }
            ],
            "agent_data": {
                "turns": [
                    {
                        "turn_index": 0,
                        "events": events
                    }
                ]
            }
        })
        
    out_data = {"eval_cases": out_cases}
    os.makedirs("artifacts/traces", exist_ok=True)
    with open("artifacts/traces/generated_traces.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("Wrote traces to artifacts/traces/generated_traces.json")

if __name__ == "__main__":
    asyncio.run(generate())

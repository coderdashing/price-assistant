import asyncio
from expense_agent.agent import app
from google.adk.events.request_input import RequestInput
import json

async def main():
    prompt = {"amount": 50, "submitter": "test", "category": "test", "description": "test", "date": "test"}
    session_id = "test_session_2"
    result = await app.run(json.dumps(prompt), session_id=session_id)
    print(f"Result: {result}")
    session = await app.session_service.get_session(session_id)
    print("Events:")
    for e in session.events:
        print(f" - {e}")
if __name__ == "__main__":
    asyncio.run(main())

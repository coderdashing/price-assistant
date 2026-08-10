import asyncio
from expense_agent.agent import app
from google.adk.runners import InMemoryRunner
from google.genai import types

runner = InMemoryRunner(app=app)

async def main():
    new_msg = types.Content(role="user", parts=[types.Part.from_text(text='{"amount": 200, "submitter": "bob@company.com", "category": "hardware", "description": "New mechanical keyboard", "date": "2026-06-19"}')])
    session_id = "debug_1"
    await runner.session_service.create_session(app_name=app.name, user_id="eval_user", session_id=session_id)
    try:
        async for event in runner.run_async(user_id="eval_user", session_id=session_id, new_message=new_msg):
            print("YIELDED EVENT TYPE:", type(event))
            print("DIR:", dir(event))
            print("REPR:", repr(event))
            print("-" * 40)
    except Exception as e:
        print("EXCEPTION THROWN:", type(e), e)

asyncio.run(main())

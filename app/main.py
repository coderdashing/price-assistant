import base64
import json
import logging
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from expense_agent.agent import app as adk_app
from google.adk.runners import InMemoryRunner
from google.genai import types

runner = InMemoryRunner(app=adk_app)

# Standard Python logging per checklist
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ambient Expense Agent")

async def process_pubsub_message(payload: dict):
    try:
        message = payload.get("message", {})
        data_b64 = message.get("data")
        
        if not data_b64:
            logger.warning("Received Pub/Sub message without 'data' field.")
            return

        # Decode the base64 payload from Pub/Sub
        data_json = base64.b64decode(data_b64).decode("utf-8")
        
        # Normalize the subscription path (e.g., "projects/myproj/subscriptions/mysub" -> "mysub")
        subscription = payload.get("subscription", "default_sub")
        short_sub_name = subscription.split("/")[-1]
        
        # Create a readable session ID from the sub name and message ID
        message_id = message.get("messageId", "unknown")
        session_id = f"{short_sub_name}-{message_id}"
        
        logger.info(f"Processing message {message_id} from {short_sub_name}. Starting session: {session_id}")
        
        # Create session if it doesn't exist
        try:
            session = await runner.session_service.get_session(session_id)
        except Exception:
            session = await runner.session_service.create_session(
                app_name=adk_app.name, user_id="pubsub_system", session_id=session_id
            )

        # Feed the decoded data into the ADK workflow
        new_message = types.Content(role="user", parts=[types.Part.from_text(text=data_json)])
        
        last_event = None
        async for event in runner.run_async(
            user_id="pubsub_system",
            session_id=session_id,
            new_message=new_message
        ):
            last_event = event
        
        logger.info(f"Workflow suspended/completed for session {session_id}.")
        
    except Exception as e:
        logger.error(f"Error processing Pub/Sub message: {e}", exc_info=True)

@app.post("/")
@app.post("/apps/expense_agent/trigger/pubsub")
async def pubsub_push(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint to receive Pub/Sub push messages.
    """
    try:
        payload = await request.json()
        # Feed into the workflow in the background to return 200 OK to Pub/Sub immediately
        background_tasks.add_task(process_pubsub_message, payload)
        return {"status": "accepted"}
    except Exception as e:
        logger.error(f"Failed to parse push payload: {e}")
        return {"status": "error", "message": "Invalid payload"}

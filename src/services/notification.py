import redis.asyncio as redis
from src.config import settings
import asyncio
import json

class NotificationService:
    def __init__(self):
        # Async Redis client for SSE streaming
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def publish_update(self, batch_id: str, message: dict):
        """Worker calls this to notify subscribers."""
        channel = f"batch:{batch_id}"
        await self.redis.publish(channel, json.dumps(message))

    async def event_generator(self, batch_id: str, request):
        """
        API calls this to yield Server-Sent Events (SSE) to the user.
        Includes handling for client disconnects.
        """
        pubsub = self.redis.pubsub()
        channel = f"batch:{batch_id}"
        await pubsub.subscribe(channel)
        
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                    
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    # Format as SSE (data: ...)
                    yield f"data: {message['data']}\n\n"
                
                # Prevent CPU spin
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
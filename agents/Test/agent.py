from agentuity import AgentRequest, AgentResponse, AgentContext
from openai import AsyncOpenAI

client = AsyncOpenAI()

def welcome():
    return {
        "welcome": "Welcome to the OpenAI Python Agent! I can help you build AI-powered applications using OpenAI models.",
        "prompts": [
            {
                "data": "How do I implement streaming responses with OpenAI models?",
                "contentType": "text/plain"
            },
            {
                "data": "What are the best practices for prompt engineering with OpenAI?",
                "contentType": "text/plain"
            }
        ]
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    try:
        result = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                    {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise and accurate information.",
                },
                {
                    "role": "user",
                    "content": await request.data.text() or "Hello, OpenAI",
                },
            ],
        )

        return response.text(result.choices[0].message.content)
    except Exception as e:
        context.logger.error(f"Error running agent: {e}")

        return response.text("Sorry, there was an error processing your request.")

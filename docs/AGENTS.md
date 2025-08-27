# Agentuity Python Agent Development

This guide provides comprehensive instructions for developing AI agents using the Agentuity platform with Python.

## 1. Agent Development Guidelines

- Prefer using the `agentuity agent create` command to create a new Agent
- Prefer importing types from the `agentuity` package
- The file should define an async function named `run`
- All code should follow Python best practices and type hints
- Use the provided logger from the `AgentContext` interface such as `context.logger.info("my message: %s", "hello")`

### Example Agent File

```python
from agentuity import AgentRequest, AgentResponse, AgentContext

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    return response.json({"hello": "world"})
```

## 2. Core Interfaces

### Agent Handler

The main handler function for an agent:

```python
async def run(
    request: AgentRequest,
    response: AgentResponse,
    context: AgentContext
) -> Any:
    # Agent implementation
    pass
```

### AgentRequest

The `AgentRequest` class provides methods for accessing request data:

- `request.trigger`: Gets the trigger type of the request
- `request.metadata`: Gets metadata associated with the request
- `request.get(key, default)`: Gets a value from the metadata
- `request.data.contentType`: Gets the content type of the request payload
- `request.data.json`: Gets the payload as a JSON object
- `request.data.text`: Gets the payload as a string
- `request.data.binary`: Gets the payload as bytes

### AgentResponse

The `AgentResponse` class provides methods for creating responses:

- `response.json(data, metadata)`: Creates a JSON response
- `response.text(data, metadata)`: Creates a text response
- `response.binary(data, content_type, metadata)`: Creates a binary response
- `response.html(data, metadata)`: Creates an HTML response
- `response.empty(metadata)`: Creates an empty response
- `response.handoff(params, args, metadata)`: Redirects to another agent
- Media-specific methods: `pdf()`, `png()`, `jpeg()`, `gif()`, `mp3()`, `mp4()`, etc.

### AgentContext

The `AgentContext` class provides access to various capabilities:

- `context.logger`: Logging functionality
- `context.kv`: Key-Value storage
- `context.vector`: Vector storage
- `context.get_agent(agent_id_or_name)`: Gets a handle to a remote agent
- `context.tracer`: OpenTelemetry tracing
- Environment properties: `sdkVersion`, `devmode`, `orgId`, `projectId`, etc.

## 3. Storage APIs

### Key-Value Storage

Access through `context.kv`:

- `await context.kv.get(name, key)`: Retrieves a value
- `await context.kv.set(name, key, value, params)`: Stores a value with optional params
- `await context.kv.delete(name, key)`: Deletes a value

### Vector Storage

Access through `context.vector`:

- `await context.vector.upsert(name, *documents)`: Inserts or updates vectors
- `await context.vector.search(name, params)`: Searches for vectors
- `await context.vector.delete(name, *ids)`: Deletes vectors

## 4. Logging

Access through `context.logger`:

- `context.logger.debug(message, *args, **kwargs)`: Logs a debug message
- `context.logger.info(message, *args, **kwargs)`: Logs an informational message
- `context.logger.warn(message, *args, **kwargs)`: Logs a warning message
- `context.logger.error(message, *args, **kwargs)`: Logs an error message
- `context.logger.child(**kwargs)`: Creates a child logger with additional context

## 5. Best Practices

- Use type hints for better IDE support
- Import types from `agentuity`
- Use structured error handling with try/except blocks
- Leverage the provided logger for consistent logging
- Use the storage APIs for persisting data
- Consider agent communication for complex workflows

For complete documentation, visit: https://agentuity.dev/SDKs/python/api-reference

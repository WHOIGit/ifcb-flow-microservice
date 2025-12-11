# Template Stateless Service

An example template stateless service. Intended to be used as starter code for services.

- `POST /echo` — Echoes the request content back to the user.

## Run with Docker Compose

```bash
docker compose up --build
```

## Request Example

```bash
curl -X POST http://localhost:8001/echo \
  -H "Content-Type: application/json" \
  -d '{
        "content": "hello service!"
      }'
```

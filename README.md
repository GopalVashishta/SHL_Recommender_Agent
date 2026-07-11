# SHL Assessment Agent

FastAPI service for grounded SHL assessment shortlisting over the catalog in `data/shl_product_catalog.json`.

## Endpoints

- `GET /health` -> `{"status":"ok"}`
- `POST /chat` -> stateless conversation replay with `reply`, `recommendations`, and `end_of_conversation`

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
```
docker run --rm -p 8260:8000 --name shl-assessment-agent-run --env-file .env.example shl-assessment-agent
```
## Notes

- The service only recommends items from the catalog.
- Packaged solutions are filtered out by default.
- Retrieval uses a lightweight FAISS index over catalog text.
- The reply text can use an LLM if you set one of these env vars: `LLM_PROVIDER=gemini` with `GEMINI_API_KEY`, or `LLM_PROVIDER=groq` with `GROQ_API_KEY`.
- Optional model overrides: `GEMINI_MODEL` and `GROQ_MODEL`.

## Environment Variables

- `LLM_PROVIDER`: `gemini`, `groq`, or unset for deterministic replies.
- `GEMINI_API_KEY`: Gemini API key.
- `GEMINI_MODEL`: defaults to `gemini-1.5-flash`.
- `GROQ_API_KEY`: Groq API key.
- `GROQ_MODEL`: defaults to `llama-3.1-8b-instant`.


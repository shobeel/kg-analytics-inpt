"""HTTP front for extraction (used by the API service in Phase 7)."""

from fastapi import FastAPI

app = FastAPI(title="extraction")


@app.get("/health")
def health() -> dict:
    return {"service": "extraction", "status": "ok"}

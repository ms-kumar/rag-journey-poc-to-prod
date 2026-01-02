from fastapi import FastAPI

from src.api.v1.endpoints import rag

app = FastAPI(title="Advanced RAG API", version="1.0.0")

# Include routers
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

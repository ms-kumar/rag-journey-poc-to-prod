from fastapi import FastAPI

app = FastAPI(title="Advanced RAG API", version="1.0.0")

@app.get("/")
def health_check():
    return {"status": "API is running"}

if __name__ == "__main__":
    # main()
    pass

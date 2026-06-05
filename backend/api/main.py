from fastapi import FastAPI


app = FastAPI(
    title="LLM Serving Platform",
    version="1.0.0",
)


@app.get("/")
def root():

    return {
        "message":
        "LLM Serving Platform"
    }
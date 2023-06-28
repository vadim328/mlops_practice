from fastapi import FastAPI
import model

app = FastAPI()

@app.get("/")
async def acc():
    return model.predict()


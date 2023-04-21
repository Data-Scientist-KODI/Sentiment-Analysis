from fastapi import FastAPI, Request
from services import sentimentAnalysis as sa
from typing import Union
app = FastAPI()


@app.get("/sentiment")
async def sentiment_analysis(request: Request):
    params = request.query_params.get("inputText")
    sentiment = await sa.sent_analysis(params)

    return {"message": str(sentiment)}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
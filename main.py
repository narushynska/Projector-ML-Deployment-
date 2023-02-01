import pickle

# import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, constr


prediction_pipeline = pickle.load(open('model/pipeline.pickle','rb'))

app = FastAPI()

class UserRequestIn(BaseModel):
    text: constr(min_length=3,max_length=1400)

class ScoredLabelsOut(BaseModel):
    score: float

@app.post("/predict", response_model=ScoredLabelsOut)
def read_classification(user_request_in: UserRequestIn):
    return {'score': prediction_pipeline.predict(pd.Series(data=[user_request_in.text]))[0].item()}
# nlp-regressor
Implement NLP model for this problem: https://www.kaggle.com/c/commonlitreadabilityprize/overview

## How to train the model
RMSE for Random forest regressor is 0.34009
### Install
This part of the project requires Python3 and the following Python libraries installed:

* pandas==1.5.3
* scikit-learn==1.2.1
* nltk==3.8.1

You can use `requirements.txt` to install them via pip.

### Code
Template code is provided in the `train.py` file. You will also be required to use the included `NLP_preprocessor.py` Python file and the `data/train.csv` dataset file to complete your work. 

Make sure your dataset path is the same as in train.py file in line 13.

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this `train.py`) and run the following command:
```sh
python train.py
```
Depending on your python settings, you also can try
```sh
python3 train.py
```
The training script will create a model folder and store preprocessing pipeline with a trained random forest regressor.

## How to run API for prediction
### Install
This part of the project requires Python3 and the following Python libraries installed:

* pandas==1.5.3
* scikit-learn==1.2.1
* nltk==3.8.1
* pydantic==1.10.4
* FastAPI==0.78.0
* uvicorn==0.18.2

You can use `requirements.txt` to install them via pip.

### Code
Template code is provided in the `main.py` file. You will also be required to use the included `NLP_preprocessor.py` Python file and the `model/pipeline.pickle` file to complete your work. 

Ensure your model path is the same as in `main.py` file in line 9.

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this `main.py`) and run the following command:
```sh
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### How to test an API

Make a POST request to `/predict` with the text you want to predict.
Here is an example using curl:
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "To explain transitivity, let us look first at a totally different example that has a lot to do with transitivity but nothing to do with rationality. A combination of choices is rational if an observer can deduce someone else preferences (that is, learn what that person likes best, medium, and least) just by observing his choices. Suppose I put your friends Adam and Ben side-by-side and I see that Adam is taller than Ben. Then I put Ben and Charlie side-by-side and I see that Ben is taller than Charlie. Can you tell me who is taller, Adam or Charlie? Adam, of course. How do we know? We simply use transitivity: if Adam is taller than Ben, and Ben is taller than Charlie, then Adam is taller than Charlie. This also means that I can rank your three friends by their height: from tallest (Adam) to shortest (Charlie)."
}'
```
## Deployed API
You can test the model result using this link https://nlp-regressor-w3jeaohkza-od.a.run.app/predict and the example above.

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union
from models import get_crop_model,get_fertilizer_model,get_input

app = FastAPI()

# Load the trained models
crop_model = get_crop_model()
fertilizer_model = get_fertilizer_model()

#--------------------------------------------------------------------

class InputArray(BaseModel):
    array: List[Union[float, str]] = Field(..., description="An array of floats or strings")

@app.post("/crop_recommend")
def array_endpoint(request: Request, input: InputArray):
    
    print("Making Crop Prediction...")
    # Make a prediction using the input array
    prediction = crop_model.predict([input.array])
    prediction = prediction.tolist()[0]
    
    print("Returning Response...")    
    # Return the prediction in the response
    return prediction


@app.post("/fertilizer_recommend")
def array_endpoint(request: Request, input: InputArray):
    
    # process the input array
    x = get_input(input.array)
    
    print("Making Fertilizer Prediction...")
    # Make a prediction using the input array
    prediction = fertilizer_model.predict([x])
    prediction = prediction[0]
    
    print("Returning Response...")    
    # Return the prediction in the response
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # changed 127.0.0.1 to 0.0.0.0 for railway.app deployment
    # you can go to "/docs" or "/redoc" endpoint to get the API documentation


## CropFusionAI Backend
This repo contains all the Backend services that support the CropFusionAI Frontend [here](https://github.com/deepeshdm/CropFusionAI).

It contains the following things : 
- Model Training Data
- Model Training Notebooks
- Trained ML models for Crop & Fertilizer Classification
- Backend FastAPI services that exposes both models through REST API's.

<div align="center"> <h3>  🔥 API Docs 👉 <a href="https://8080-797137136eb6451193a1f8c64a951490.patr.cloud/docs"> here </a> <div align="center"> </h3> </div>
<div align="center">
<img src="/data/fapi.png" width="95%"/>
</div>


## Code Examples
Below are some code examples you can utilize to send post requests to different endpoints with different payloads to make the most out of the ML models.

### 1] Crop Recommendation API

```python
import requests
url = "https://8080-797137136eb6451193a1f8c64a951490.patr.cloud/crop_recommend"
payload = { "array": [55,44,33,40,75,6.5,300] }
response = requests.post(url, json=payload)
print(response.json())
```

### 2] Fertilizer Recommendation API

```python
import requests
url = "https://8080-797137136eb6451193a1f8c64a951490.patr.cloud/fertilizer_recommend"
payload = { "array": [33,56,30,88,91,12,"Sandy","Cotton"] }
response = requests.post(url, json=payload)
print(response.json())
```

## Links to Resources
- Crop Recommendation Dataset [here](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Fertilizer Recommendation Dataset [here](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)

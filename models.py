
import numpy as np
import lightgbm as lgb
import pandas as pd

# Train and Return Crop recommendation model
def get_crop_model():
    
    # Loading the downloaded dataset
    path = r"data/Crop_recommendation.csv"
    df = pd.read_csv(path)

    print("Loading crop dataset...")
    x = df.drop("label", axis=1)
    y = df["label"]

    model = lgb.LGBMClassifier()

    # Training the model using Training Data
    print("Training crop model...")
    model.fit(x,y)
    
    return model
    
#-----------------------------------------------------------------------------------------

# Train and Return Fertilizer recommendation model
def get_fertilizer_model():
    
    print("Loading fertilizer dataset...")
    # Loading the downloaded dataset
    df = pd.read_csv(r"data/Fertilizer_Prediction.csv")
    # rename target column
    df = df.rename({'Fertilizer Name': 'Fertilizer','Crop Type': 'Crop_Type','Soil Type': 'Soil_Type'}, axis=1)

    #-----------------------------------------------

    # one-hot encoding

    print("One-hot encoding...")
    # list of categorical features in dataset
    categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']
    # Remove the Target variable.
    categorical_features.remove('Fertilizer')

    # encode categorical features
    new_encoded_columns = pd.get_dummies(df[categorical_features])
    # Concatinating with original dataframe
    df = pd.concat([df,new_encoded_columns],axis="columns")

    # dropping the categorical variables since they are redundant now.
    df = df.drop(categorical_features,axis="columns")

    #-----------------------------------------------

    print("Preparing x and y...")
    x = df.drop("Fertilizer",axis=1)
    y = df["Fertilizer"]

    #-----------------------------------------------

    print("Data spliting...")

    # DATA SPLITTING 
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,shuffle=True)

    #-----------------------------------------------

    print("Training the fertilizer model...")

    # Creating a lightgbm model
    import lightgbm as lgb

    model = lgb.LGBMClassifier()

    # Training the model using Training Data
    model.fit(x_train,y_train)
    
    return model

#-----------------------------------------------------------------------------------------

def get_input(x):

# Index values of each variable in x
    x_structure = {
        "Temparature": 0, "Humidity": 1, "Moisture": 2, "Nitrogen": 3,
        "Potassium": 4, "Phosphorous": 5, "Black": 6,  "Clayey": 7, "Loamy": 8,
        "Red": 9, "Sandy": 10, "Barley": 11, "Cotton": 12, "Ground Nuts": 13, "Maize": 14,
        "Millets": 15, "Oil seeds": 16, "Paddy": 17, "Pulses": 18, "Sugarcane": 19, "Tobacco": 20,
        "Wheat": 21
    }

    output = np.zeros(len(x_structure))
    output[0] = x[0]
    output[1] = x[1]
    output[2] = x[2]
    output[3] = x[3]
    output[4] = x[4]
    output[5] = x[5]
    output[x_structure[x[6]]] = 1
    output[x_structure[x[7]]] = 1
    return output

#-----------------------------------------------------------------------------------------

# print("Making crop prediction...")
# model = get_crop_model()
# output = model.predict([[59,66,47.77,32.555,90.1,5.7932454170000005,233.0745066]])
# print("Predicted Crop : ",output)

# print("Running the input_function...")
# x1 = get_input([25,50,26,15,14,11,"Red","Ground Nuts"])
# model = get_fertilizer_model()
# y1 = model.predict([x1])
# print("Predicted Fertilizer : ",y1[0])


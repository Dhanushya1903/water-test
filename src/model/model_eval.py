import numpy as np
import pandas as pd

import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception (f"Error loading data from {filepath} : {e}")
#test_data = pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        x = data.drop(columns=['Potability'],axis = 1)
        y = data['Potability']
        return x,y
    except Exception as e:
        raise Exception(f"Error Preparing data: {e}")

#x_test = test_data.iloc[:,0:-1].values
#y_test = test_data.iloc[:,-1].values

def load_model(filepath : str):
    try:
        with open(filepath,"rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception (f"Error loading model from {filepath} : {e}")

#model = pickle.load(open("model.pkl","rb"))

def evaluation_model(model, x_test : pd.DataFrame, y_test : pd.Series) ->dict:
    try:
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)

        metrics_dict = {
            'accuracy':acc,
            'precision':pre,
            'recall':recall,
            'f1_score':f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception (f"Error evaluating model : {e}")



def save_metrics(metrics_dict : dict, filepath : str) -> None:
    try:
        with open(filepath, 'w') as file:  # ✅ use the parameter
            json.dump(metrics_dict, file, indent=4)

    except Exception as e:
        raise Exception (f"Error saving metrics to {filepath} : {e}")
    

def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "model.pkl"
        metrics_path = "reports/metrics.json"


        test_data = load_data(test_data_path)
        x_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,x_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception (f"An error occured : {e}")
    
if __name__ == "__main__":
    main()

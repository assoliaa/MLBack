import json
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd 
import numpy as np

app =FastAPI()

class Stroke(BaseModel):
     age:int
     hypertension:int 
     heart_disease:int 
     avg_glucose_level:float
     bmi:float
     smoking_status:int   
     Female:int  
     Male:int                
     not_married:int      
     married : int           
     Rural:int               
     Urban:int              


class Pcos(BaseModel):
     Age:int
     Weight:float
     Height_cm:float
     BMI:float                         
     Blood_Group:int             
     Pulse_rate:int      
     Breath_min:int          
     Hb_g_dl:float                   
     Cycle_regularity: int             
     Cycle_length_days: int      
     Marraige_Status_years:int
     Pregnancy:int
     FSH_mIU_mL:float
     LH_mIU_mL:float
     FSH_LH:float
     Hip_inch:int         
     Waist_inch:int         
     Waist_Hip_Ratio:float        
     TSH_mIU_L:float             
     AMH_ng_mL:float              
     PRL_ng_mL:float           
     Vit_D3_ng_mL:float                  
     Weight_Gain_Y_N: int      
     Hair_growth_Y_N: int      
     Skin_darkening: int   
     Hair_loss_Y_N:  int        
     Pimples:  int         
     Fast_food_Y_N: int        
     Reg_Exercise_Y_N: int      
     BP_Systolic_mmHg: int    
     BP_Diastolic_mmHg:int   
     Follicle_No_L: int       
     Follicle_No_R: int       
     Avg_F_size_L_mm: float   
     Avg_F_size_R_mm: float  
     Endometrium_mm:  float  

class Diabetes(BaseModel):
     age:float
     hypertension:int 
     heart_disease:int
     smoking_history:int
     bmi:float 
     HbA1c_level:float
     blood_glucose_level:int   
     Female:int  
     Male:int                  

def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        print(f"Model file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None

pcos_model_path = 'pcos_model.pkl'
diabetes_model_path = 'diabetes_model.pkl'
stroke_model_path = 'stroke_model.pkl'

pcos = load_model(pcos_model_path)
diabetes = load_model(diabetes_model_path)
stroke = load_model(stroke_model_path)

def predict_chances(item, model):
    try:
        data=item.json()
        data_dict = json.loads(data)
        li =list(data_dict.values())
        prediction = model.predict([li])
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.post('/pcos')
async def pcos_endpoint(item: Pcos):
    prediction = predict_chances(item, pcos)
    if (prediction[0] == 1):
        return "Chances of having PCOS are quite high"
    else:
        return "Chances of having PCOS are quite low"

@app.post('/stroke')
async def stroke_endpoint(item: Stroke):
    prediction = predict_chances(item, stroke)
    if (prediction[0] == 1):
        return "Chances of having stroke are quite high"
    else:
        return "Chances of having stroke are quite low"

@app.post('/diabetes')
async def diabetes_endpoint(item: Diabetes):
    prediction = predict_chances(item, diabetes)
    if (prediction[0] == 1):
        return "Chances of having diabetes are quite high"
    else:
        return "Chances of having diabetes are quite low"


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load


subject_list=[
    "Mathematics"
,"Physics"
,"Chemistry"
,"Biology"
,"Computer Science"
,"Accountancy"
,"Economics"
,"Business Studies"
,"Literature"
,"History"
,"Geography"
,"Sociology"
,"Fine Arts"
,"Political Science"
,"Legal Studies"
,"English"
,"Media Studies"
,"Communication"
,"Music"
,"Dance"
,"Drama/Theatre"
,"Aeronautics"
,"Environmental Science"
,"Sports Science"
,"Psychology"
,"Home Science"
,"Physical Education"
,"Anatomy"
,"Food Science"
,"Library Science"
,"Information Technology"
,"Library Science"
,"Marine Science"
,"Forensic Science"
]

subject_dict={}

df = pd.DataFrame()

def making_df():    
    for subject in subject_list:
        subject_dict[subject]=[None]
    subject_dict["Domain"]=[None]
    subject_dict["Prefered Domain"]=[None]
    global df
    df=pd.DataFrame(subject_dict)

def preprocessor_1(subs):
  i=0
  record={}
  for sub in subs:
    its=sub.lstrip()
    its=sub.rstrip()
    record[sub]=int(1)
  df.iloc[0]=record
#   return record

def preprocessor_2():   
   df_filled = df.fillna(0)
   columns_to_convert = df_filled.columns 
   df_filled[columns_to_convert] = df_filled[columns_to_convert].astype('int64')
   inp1=df_filled.loc[0]
   inp_lst=[inp1[:len(inp1)-2]]
   return inp_lst

def prediction(inp_lst):
   loaded_model = load('decision_tree_model.joblib')
   prediction_result=loaded_model.predict(inp_lst)
   return prediction_result

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the POST request
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Invalid input. Expected a JSON list of strings.'}), 400
    
    making_df()
    preprocessor_1(data)
    res=prediction(preprocessor_2())
    print(res)
    # Process the list of strings (In this example, just concatenating them)
    result = ' '.join(data)
    
    # Return the result in JSON format
    ans=res[0]
    return jsonify({'result': ans})

if __name__ == '__main__':
    app.run(debug=True)


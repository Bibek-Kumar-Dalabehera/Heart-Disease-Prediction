from flask  import Flask,render_template,request
import joblib
import pandas as pd


app=Flask(__name__)

data = pd.read_csv("heart.csv")


import joblib
import pandas as pd

def model(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    input_data = [[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]]

    # Load the encoders
    chest_pain_encoder = joblib.load('ChestPainTypeEncode.pkl')  # Encoder for ChestPainType
    resting_ecg_encoder = joblib.load('RestingECGEncode.pkl')    # Encoder for RestingECG

    # Create the input DataFrame
    input_df = pd.DataFrame(input_data, columns=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", 
                                                 "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"])

    # Encoding
    input_df.Sex.replace(
    {
        'M' : 0,
        'F' : 1
    },
    inplace=True
    )
    input_df.ExerciseAngina.replace(
        {
            'N': 0,
            'Y': 1
        },
        inplace=True
    )
    input_df.ST_Slope.replace(
        {
            'Down': 1,
            'Flat': 2,
            'Up': 3
        },
        inplace=True
    )
    
    # Apply one-hot encoding to ChestPainType
    chest_pain_encoded = chest_pain_encoder.transform(input_df[['ChestPainType']])
    chest_pain_encoded_df = pd.DataFrame(chest_pain_encoded, columns=chest_pain_encoder.get_feature_names_out(['ChestPainType']), index=input_df.index)

    # Apply one-hot encoding to RestingECG
    resting_ecg_encoded = resting_ecg_encoder.transform(input_df[['RestingECG']])
    resting_ecg_encoded_df = pd.DataFrame(resting_ecg_encoded, columns=resting_ecg_encoder.get_feature_names_out(['RestingECG']), index=input_df.index)

    # Drop the original columns and merge the encoded columns
    input_df = input_df.drop(columns=['ChestPainType', 'RestingECG']).reset_index(drop=True)
    input_df = pd.concat([input_df, chest_pain_encoded_df, resting_ecg_encoded_df], axis=1)


    # # Prediction using the model
    model_file = joblib.load('Heart_Desease_model.pkl')
    probabilities = model_file.predict_proba(input_df)
    heart_disease_prob = probabilities[0][1]  # Probability of heart disease (class 1)
    return round(heart_disease_prob * 100, 2)

    

@app.route('/')
def index():
    return render_template('index.html') 

# About section
@app.route('/about')
def home():
        return render_template('about.html')  

# Login Page
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

    return render_template('login.html')

#Signup page
@app.route('/signupform',methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
    return render_template('signup.html')

# Pre Predict page
@app.route('/prepredict',methods=['GET', 'POST'])
def prepredict():
    return render_template('prepredict.html')

# Predict page
@app.route('/prediction',methods=['GET', 'POST'])
def prediction():
    Sex=sorted(data['Sex'].unique())
    ChestPainType=sorted(data['ChestPainType'].unique())
    RestingECG=sorted(data['RestingECG'].unique())
    ExerciseAngina=sorted(data['ExerciseAngina'].unique())
    ST_Slope=sorted(data['ST_Slope'].unique())

    return render_template('prediction.html',Sex=Sex, ChestPainType= ChestPainType,RestingECG=RestingECG, ExerciseAngina=ExerciseAngina, ST_Slope=ST_Slope)

@app.route('/price', methods= ['GET','POSt'])
def price_pred():
        
        Age = request.form.get('Age')
        Sex = request.form.get('Sex')
        ChestPainType = request.form.get('ChestPainType')
        RestingBP = request.form.get('RestingBP')
        Cholesterol = request.form.get('Cholesterol')
        FastingBS = request.form.get('FastingBS')
        RestingECG = request.form.get('RestingECG')
        MaxHR = request.form.get('MaxHR')
        ExerciseAngina =  request.form.get('ExerciseAngina')
        Oldpeak = request.form.get('Oldpeak')
        ST_Slope = request.form.get('ST_Slope')

        # Make Prediction
        heart_disease_prob = model(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)

        # Return result as percentage
        return str(heart_disease_prob) + '%'


    

if __name__ =='__main__':
    app.run(debug=True,port=7069)
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('trick_play_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    # Initial render without predictions
    return render_template('index.html', prediction_text="", down="", yardline_100="", ydstogo="", epa="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from the form
        features = {
            'down': [int(request.form['down'])],
            'yardline_100': [int(request.form['yardline_100'])],
            'ydstogo': [int(request.form['ydstogo'])],
            'epa': [float(request.form['epa'])]  # Accept EPA as a float
        }
        input_data = pd.DataFrame(features)

        # Predict using the trained model
        prediction = model.predict(input_data)[0]
        
        # Return with populated values to retain form data
        return render_template(
            'index.html', 
            prediction_text=f"Punt Trick Play Probability: {prediction * 100:.2f}%",
            down=request.form['down'],
            yardline_100=request.form['yardline_100'],
            ydstogo=request.form['ydstogo'],
            epa=request.form['epa']
        )
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

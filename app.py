from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('trick_play_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', prediction_text="", yardline_100="", ydstogo="", time_remaining="", lions_score="", opponent_score="", wind="", quarter="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        lions_score = int(request.form['lions_score'])
        opponent_score = int(request.form['opponent_score'])
        score_differential = lions_score - opponent_score
        
        yardline_100 = int(request.form['yardline_100'])
        ydstogo = int(request.form['ydstogo'])
        time_remaining = int(request.form['time_remaining'])
        quarter = int(request.form['quarter'])
        wind = float(request.form['wind'])
        
        # Convert time_remaining to minutes:seconds format
        minutes = time_remaining // 60
        seconds = time_remaining % 60
        formatted_time = f"{minutes}:{seconds:02d}"
        
        # Convert quarter to ordinal representation
        quarter_suffix = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        formatted_quarter = quarter_suffix.get(quarter, f"{quarter}th")
        
        features = {
            'yardline_100': [yardline_100],
            'ydstogo': [ydstogo],
            'time_remaining': [float(time_remaining)],
            'score_differential': [float(score_differential)],
            'wind': [wind],
            'quarter': [quarter]
        }
        input_data = pd.DataFrame(features)

        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Render results
        return render_template(
            'index.html',
            prediction_text=f"Punt Trick Play Probability: {prediction * 100:.2f}%",
            yardline_100=yardline_100,
            ydstogo=ydstogo,
            time_remaining=formatted_time,  # Pass formatted time
            lions_score=lions_score,
            opponent_score=opponent_score,
            wind=wind,
            quarter=formatted_quarter  # Pass formatted quarter
        )
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

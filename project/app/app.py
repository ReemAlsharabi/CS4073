from flask import Flask, render_template
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Load the preprocessed data
    df = pd.read_csv('wrangled_data.csv')

    # Create visualizations
    fig1 = px.line(df, x='dteday', y='cnt', title='Total Bike Rentals Over Time')
    fig2 = px.box(df, x='season', y='cnt', color='weathersit', title='Bike Rentals by Season and Weather')
    fig3 = px.scatter(df, x='temp', y='cnt', color='season', trendline='ols', title='Bike Rentals vs. Temperature')

    # Convert the visualizations to HTML
    plot1 = fig1.to_html(full_html=False)
    plot2 = fig2.to_html(full_html=False)
    plot3 = fig3.to_html(full_html=False)

    return render_template('dashboard.html', plot1=plot1, plot2=plot2, plot3=plot3)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the preprocessed data
    df = pd.read_csv('wrangled_data.csv')

    # Split the data into features and target
    X = df.drop('cnt', axis=1)
    y = df['cnt']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the root mean squared error
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Save the trained model
    joblib.dump(model, 'bike_share_model.pkl')

    return render_template('index.html', prediction=rmse)

if __name__ == '__main__':
    app.run(debug=True)
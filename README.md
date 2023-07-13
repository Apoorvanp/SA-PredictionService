# BeeHive-PredictionService
Prediction Service for SA Beehive Project

The forecasting system is trained on historical solar radiation data.

Build the Docker image using the provided Dockerfile:

``` docker build -t prediction_beehive:latest2 . ```

Docker run 

``` docker run -d -p 3000:3000 myapp:latest ```


Other solution

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment:
Run ``` python3 -m venv env ``` to create a new virtual environment.
4. Activate the virtual environment:
On macOS and Linux: ``` source env/bin/activate ```
On Windows: ``` .\env\Scripts\activate ```
5. Install the required dependencies:
Run ``` pip install -r requirements.txt ``` to install the necessary packages.


Run the application:
Use ```python3 hourly_prediction.py`` or ```python hourly_prediction.py``` to start the Flask application - Hourly prediction.
Use ```python3 monthly_prediction.py`` or ```python monthly_prediction.py``` to start the Flask application - Monthly prediction.

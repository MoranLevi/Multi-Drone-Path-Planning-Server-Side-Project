from flask import Flask, request, jsonify
import algorithm
from flask_cors import CORS
from waitress import serve

# Create the application instance with flask 
app = Flask(__name__)
# Configure CORS
CORS(app)


# Create a URL route in our application for "/optimal-targets-classification with get request"
@app.route('/optimal-targets-classification', methods=['GET'])
def get_optimal_numbers_of_drones():
    print("Request received for optimal targets classification\n")
    fileName = request.args.get('fileName') # Get the file name from the request
    numbersOfDrones = -1 # Set the number of drones to -1, as this indicates the optimal number of drones
    results = algorithm.start_algorithm(fileName, numbersOfDrones) # Call the algorithm with the file name and number of drones
    return jsonify(results) # Return the results


# Create a URL route in our application for "/required-targets-classification with get request"
@app.route('/required-targets-classification', methods=['GET'])        
def get_required_targets_classification():
    print("Request received for required targets classification\n")
    fileName = request.args.get('fileName') # Get the file name from the request
    numberOfDrones = request.args.get('numberOfDrones') # Get the number of drones from the request
    results = algorithm.start_algorithm(fileName, numberOfDrones) # Call the algorithm with the file name and number of drones
    return jsonify(results) # Return the results


# main function
if __name__ == '__main__':
    print("Starting server...\n")
    serve(app, host='localhost', port=8000) # Run the application on port 8000
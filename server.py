from flask import Flask, Response,request, jsonify
from http.server import HTTPServer, BaseHTTPRequestHandler
import algorithm
from flask_cors import CORS, cross_origin

numbersOfDrones = -1
fileName = None

app = Flask(__name__)
CORS(app)

@app.route('/optimal-targets-classification', methods=['GET'])
def get_optimal_numbers_of_drones():
    # fileName="TSP100.txt"
    fileName = request.args.get('fileName')
    numbersOfDrones = -1
    results = algorithm.start_algorithm(fileName, numbersOfDrones)
    # results = [1, 2, 3, 4, 5]
    return jsonify(results)
    #return results
    # return Response(f'Accepted - {fileName}', status=202, mimetype='application/json')

@app.route('/required-targets-classification', methods=['GET'])        
def get_required_targets_classification():
    fileName = request.args.get('fileName')
    numberOfDrones = request.args.get('numberOfDrones')
    results = algorithm.start_algorithm(fileName, numberOfDrones)
    # results = [1, 2, 3, 4, 5]
    return jsonify(results)
    # return Response('Accepted', status=202, mimetype='application/json')
  

if __name__ == '__main__':
    app.run('localhost', '8000')
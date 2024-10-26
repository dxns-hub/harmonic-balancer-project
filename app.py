import socket
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from harmonic_balancer import HarmonicBalancer

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    num_qubits = data.get('num_qubits', 4)
    max_iterations = data.get('max_iterations', 1000)
    harmony_memory_size = data.get('harmony_memory_size', 20)

    def custom_objective_function(vector, param):
        return np.prod(vector) * param

    balancer = HarmonicBalancer(num_qubits=num_qubits, max_iterations=max_iterations, harmony_memory_size=harmony_memory_size, objective_function=custom_objective_function)
    best_solution, best_score = balancer.run_experiment()

    return jsonify({
        'best_solution': best_solution.tolist(),
        'best_score': best_score
    })

@app.route('/run_tests', methods=['GET'])
def run_tests():
    try:
        result = subprocess.run(['python', '-m', 'unittest', 'discover', 'tests'], capture_output=True, text=True)
        return jsonify(result.stdout)
    except Exception as e:
        return jsonify(str(e)), 500

if __name__ == "__main__":
    port = find_free_port()
    print(f'Flask app is running on http://127.0.0.1:{port}')
    app.run(debug=True, port=port, use_reloader=False)


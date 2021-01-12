from __future__ import annotations
from Matrix import Matrix
import pickle
import math

def sigmoid(x: int) -> float:
  """Calculate the sigmoid function

  Args:
      x (int): the x value for the sigmoid function

  Returns:
      float: f(x) of the sigmoid function
  """
  return 1 / (1 + math.exp(-1 * x))

def derivative_sigmoid(y: int) -> float:
  """Calculate the derivative of the sigmoid function 

  Args:
      y (int): the output of the sigmoid function of x

  Returns:
      float: the output of the derivative sigmoid function
  """
  return y * (1 - y)

class NeuralNetwork:
  def __init__(self, input_count: int, hidden_count: int, output_count: int) -> NeuralNetwork:
    """Create a Neural Network instance 

    Args:
        input_count (int): number of inputs nodes
        hidden_count (int): number of hidden nodes
        output_count (int): number of output nodes

    Returns:
        NeuralNetwork: new instance of Neural Network 
    """
    self.input_nodes = input_count;
    self.hidden_nodes = hidden_count;
    self.output_nodes = output_count;

    self.weights_input_hidden = Matrix(self.hidden_nodes, self.input_nodes)
    self.weights_input_hidden.randomize()

    self.weights_hidden_output = Matrix(self.output_nodes, self.hidden_nodes)
    self.weights_hidden_output.randomize()
    
    self.bias_hidden = Matrix(self.hidden_nodes, 1)
    self.bias_hidden.randomize()

    self.bias_output = Matrix(self.output_nodes, 1)
    self.bias_output.randomize()

    self.learning_rate = 0.01

  def __str__(self) -> str:
    retStr = "input_nodes: {}\nhidden_nodes: {}\noutput_nodes: {}\nlearning_rate: {}\n".format(self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate)
    
    return retStr

  def feedforward(self, inputs: list[int]) -> list[float]:
    """Feedforward the neural network

    Args:
        input (list[int]): the inputs for the neural network

    Returns:
        list[float]: the output of the neural network
    """
    input_matrix = Matrix.fromList(inputs)

    # Generating the hidden layer 
    hidden_matrix = Matrix.dot(self.weights_input_hidden, input_matrix)
    hidden_matrix = Matrix.add_matrix(hidden_matrix, self.bias_hidden)
    hidden_matrix.map(sigmoid)

    # Generating the outputs
    output_matrix = Matrix.dot(self.weights_hidden_output, hidden_matrix)
    output_matrix = Matrix.add_matrix(output_matrix, self.bias_output)
    output_matrix.map(sigmoid)

    return output_matrix.toList()

  def train(self, inputs: list[int], targets: list[int]) -> None:
    """Train the neural network using inputs and expected targets

    Args:
        inputs (list[int]): the inputs
        targets (list[int]): the expected outputs
    """
    # Feedforward the inputs
    input_matrix = Matrix.fromList(inputs)

    # Generating the hidden layer 
    hidden_matrix = Matrix.dot(self.weights_input_hidden, input_matrix)
    hidden_matrix = Matrix.add_matrix(hidden_matrix, self.bias_hidden)
    hidden_matrix.map(sigmoid)

    # Generating the outputs
    output_matrix = Matrix.dot(self.weights_hidden_output, hidden_matrix)
    output_matrix = Matrix.add_matrix(output_matrix, self.bias_output)
    output_matrix.map(sigmoid)
   
    # Calculate the output errors
    targets_matrix = Matrix.fromList(targets) 
    output_errors = Matrix.subtract_matrix(targets_matrix, output_matrix)

    # Calculate output gradient
    output_matrix.map(derivative_sigmoid)
    output_matrix = Matrix.multiply_matrix(output_matrix, output_errors)
    output_matrix.scale(self.learning_rate)
    
    # Calculate hidden -> outputs deltas 
    hidden_transpose = Matrix.transpose(hidden_matrix)
    weights_hidden_output_deltas = Matrix.dot(output_matrix, hidden_transpose)

    # Adjust the weights and biases
    self.weights_hidden_output = Matrix.add_matrix(self.weights_hidden_output, weights_hidden_output_deltas)
    self.bias_output = Matrix.add_matrix(self.bias_output, output_matrix)

    # Calculate the hidden error
    weights_hidden_output_transpose = Matrix.transpose(self.weights_hidden_output)
    hidden_errors = Matrix.dot(weights_hidden_output_transpose, output_errors)
    
    # Calculate hidden gradient
    hidden_matrix.map(derivative_sigmoid)
    hidden_matrix = Matrix.multiply_matrix(hidden_matrix, hidden_errors)
    hidden_matrix.scale(self.learning_rate)

    # Calculate inputs -> hiddens deltas 
    inputs_transpose = Matrix.transpose(input_matrix)
    weights_input_hidden_deltas = Matrix.dot(hidden_matrix, inputs_transpose)

    # Adjust the weights and biases
    self.weights_input_hidden = Matrix.add_matrix(self.weights_input_hidden, weights_input_hidden_deltas)
    self.bias_hidden = Matrix.add_matrix(self.bias_hidden, hidden_matrix)

  def save(self, filename: str) -> None:
    """saves the neural network instance

    Args:
        filename (str): filename to save to
    """
    with open(filename, "wb") as save_file:
      pickle.dump(self, save_file)

  @staticmethod
  def load(filename: str) -> NeuralNetwork:
    """loads the neural network save

    Args:
        filename (str): filename to load from

    Returns:
        NeuralNetwork: the new instance from the saved file
    """
    with open(filename, "rb") as load_file:
      return pickle.load(load_file)
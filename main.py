import sys
from typing import Tuple
sys.path.append('./NeuralNetwork/')
from os import path
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import mnist

def train_nn(brain: NeuralNetwork) -> None:
  """Parsing the train dataset and start training the model

  Args:
      brain (NeuralNetwork): the model to train
  """
  train_labels, train_images = mnist.load_dataset(mnist.files_type["train"])
  print("start training...")

  for i in range(len(train_images)):
    print("training {} cases".format(i))

    targets = [0]*10
    targets[train_labels[i]] = 1
    brain.train(mnist.flat_matrix(train_images[i]), targets)

def test_nn(brain: NeuralNetwork) -> Tuple[float, list[list[int, int, list[list[int]]]]]:
  """Parsing the test dataset and testing on the model

  Args:
      brain (NeuralNetwork): the model to test

  Returns:
      Tuple[float, list[list[int, int, list[list[int]]]]]: accuracy of the model, list of list of [the model result, the actual result, the image]
  """
  test_labels, test_images = mnist.load_dataset(mnist.files_type["test"])

  hit = 0
  results = []
  for i in range(len(test_images)):
    output = brain.feedforward(mnist.flat_matrix(test_images[i]))
    if output.index(max(output)) == test_labels[i]:
      hit += 1
    results.append([output.index(max(output)), test_labels[i], test_images[i]])

    if i > 0:
      print("{}: accurate: {}%".format(i, (hit/i)*100))

  return (hit / len(test_images)) * 100, results

def main():
  filename_save = "nn_save.pkl"

  if path.exists(filename_save):
    brain = NeuralNetwork.load(filename_save)
  else:
    brain = NeuralNetwork(784, 30, 10) 
  
  print(brain)
  
  accuracy, results = test_nn(brain)
  
  print("accuracy: {}%".format(accuracy))
  print("what do you want to see?")
  while choice := int(input()):
    if choice == -1:
      break
    
    fig = plt.figure()
    plt.imshow(results[choice][2],cmap = plt.cm.gray_r)
    txt = "This is: {}\nthe model thought: {}".format(results[choice][1], results[choice][0])
    fig.text(0.1,0.1,txt)
    plt.show()

  brain.save(filename_save)


if __name__ == "__main__":
    main()
from typing import Union
import numpy as np
from PIL import Image

files_type = {
  "train": ["./dataset/train-images.idx3-ubyte","./dataset/train-labels.idx1-ubyte"],
  "test": ["./dataset/t10k-images.idx3-ubyte","./dataset/t10k-labels.idx1-ubyte"],
}

def load_dataset(files: list[str]) -> Union[list[int], list[list[list[int]]]]:
  """load the images and labels of the test dataset

  Args:
      files (list[str]): list of files path for images and label dataset

  Returns:
      Union[list[int], list[list[list[int]]]]: list of labels and list of int matrixes
  """
  print("loading the dataset...")

  with open(files[0], "rb") as image_file:
    megic_number = int.from_bytes(image_file.read(4), 'big', signed=True)
    number_of_images = int.from_bytes(image_file.read(4), 'big', signed=True)
    rows = int.from_bytes(image_file.read(4), 'big', signed=True)
    cols = int.from_bytes(image_file.read(4), 'big', signed=True)
    images = []

    for _ in range(number_of_images):
      matrix = []
      for _ in range(rows):
        row = []
        for _ in range(cols):
          row.append(int.from_bytes(image_file.read(1), 'big', signed=False))
        matrix.append(row)
      images.append(matrix)
    
  with open(files[1], "rb") as label_file:
    megic_number = int.from_bytes(label_file.read(4), 'big', signed=True)
    number_of_labels = int.from_bytes(label_file.read(4), 'big', signed=True)
    labels = []

    for _ in range(number_of_labels):
      labels.append(int.from_bytes(label_file.read(1), 'big', signed=False))

  return labels, images

def flat_matrix(matrix: list[list[int]]) -> list[int]:
  """flatern matrix to a list

  Args:
      matrix (list[list[int]]): matrix to flatern

  Returns:
      list[int]: result of matrix flatern
  """
  return [item/255 for sublist in matrix for item in sublist]
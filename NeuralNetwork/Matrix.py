from __future__ import annotations
import random

class Matrix:
  def __init__(self, rows: int, cols: int) -> Matrix:
    """Create new instance of Matrix

    Args:
        rows (int): number of rows
        cols (int): number of cols

    Returns:
        Matrix: new instance of Matrix
    """
    self.rows = rows
    self.cols = cols
    self.matrix = []

    for i in range(self.rows):
      self.matrix.append([0] * self.cols)

  def __str__(self) -> str:
    """Create a string to describe the matrix 

    Returns:
        str: string that describes the matrix
    """
    retStr = "rows: {}\ncols: {}\n".format(self.rows, self.cols)
    retStr += "\n".join(list(map(lambda row: str(row), self.matrix)))

    return retStr

  def randomize(self) -> None:
    """Randomize the matrix values
    """
    self.matrix = list(map(lambda row: [random.uniform(-1, 1) for i in range(self.cols)], self.matrix))


  def scale(self, n: int) -> None:
    """Multiply each value in the matrix by n

    Args:
        n (int): the number to multiply by each value of the matrix
    """
    self.map(lambda x: x * n)

  def add_number(self, n: int) -> None:
    """Add each value in the matrix by n

    Args:
        n (int): the number to add by each value of the matrix
    """
    self.map(lambda x: x + n)

  def map(self, func: function) -> None:
    """Executes a function for each of the matrix values

    Args:
        func (function): the function to execute on each value
    """
    self.matrix = list(map(lambda row: list(map(lambda val: func(val), row)), self.matrix))

  def toList(self) -> list[int]:
    """return a list from the matrix

    Returns:
        list[int]: the new list from the matrix
    """
    list = []
  
    for i in range(self.rows):
      for j in range(self.cols):
        list.append(self.matrix[i][j])

    return list

  def setMatrix(self, matrix: list[list[int]]) -> None:
    if len(matrix) == self.rows and len(matrix[0]) == self.cols:
      self.matrix = matrix
    else:
      raise ValueError("The matrix need to be {}*{}".format(self.rows, self.cols))

  @staticmethod
  def fromList(list: list[int]) -> Matrix:
    """Create a new Matrix from a list, every value per row

    Args:
        list (list[int]): the list to turn into Matrix

    Returns:
        Matrix: the new Matrix from the list
    """
    matrix = Matrix(len(list), 1)

    for i in range(len(list)):
      matrix.matrix[i][0] = list[i]

    return matrix

  @staticmethod
  def multiply_matrix(first: Matrix, other: Matrix) -> Matrix:
    """Multiply each value of the first matrix by each value of the other matrix

    Args:
        first (Matrix): the first matrix to be multiply from the other matrix
        other (Matrix): the matrix to multiply each value to the first matrix

    Raises:
        ArithmeticError: if the matrix sizes are not equals

    Returns:
        Matrix: the multiplication of the two matrixes
    """
    if first.rows == other.rows and first.cols == other.cols:
      result = Matrix(first.rows, first.cols)
      result.matrix = list(map(lambda row, other_row: list(map(lambda val, other_val: val * other_val, row, other_row)), first.matrix, other.matrix))

      return result
    else:
      raise ArithmeticError("the two matrixes need to be the same size: {}*{} {}*{}"
      .format(len(first.matrix), len(first.matrix[0]), len(other.matrix), len(other.matrix[0])))

  @staticmethod
  def add_matrix(first: Matrix, other: Matrix) -> Matrix:
    """Add each value of the first matrix by each value of the other matrix

    Args:
        first (Matrix): the first matrix to be added from the other matrix
        other (Matrix): the matrix to add each value to the first matrix

    Raises:
        ArithmeticError: if the matrix sizes are not equals

    Returns:
        Matrix: the addition of the two matrixes
    """
    if first.rows == other.rows and first.cols == other.cols:
      result = Matrix(first.rows, first.cols)
      result.matrix = list(map(lambda row, other_row: list(map(lambda val, other_val: val + other_val, row, other_row)), first.matrix, other.matrix))

      return result
    else:
      raise ArithmeticError("the two matrixes need to be the same size: {}*{} {}*{}"
      .format(len(first.matrix), len(first.matrix[0]), len(other.matrix), len(other.matrix[0])))

  @staticmethod
  def subtract_matrix(first: Matrix, other: Matrix) -> Matrix:
    """Subtract each value of the first matrix by each value of the other matrix

    Args:
        first (Matrix): the first matrix to be subtracted from the other matrix
        other (Matrix): the matrix to subtract each value to the first matrix

    Raises:
        ArithmeticError: if the matrix sizes are not equals

    Returns:
        Matrix: the subtraction of the two matrixes
    """
    if first.rows == other.rows and first.cols == other.cols:
      result = Matrix(first.rows, first.cols)
      result.matrix = list(map(lambda row, other_row: list(map(lambda val, other_val: val - other_val, row, other_row)), first.matrix, other.matrix))

      return result
    else:
      raise ArithmeticError("the two matrixes need to be the same size: {}*{} {}*{}"
      .format(len(first.matrix), len(first.matrix[0]), len(other.matrix), len(other.matrix[0])))

  @staticmethod
  def transpose(matrix: Matrix) -> Matrix:
    """Transpose the matrix

    Args:
        matrix (Matrix): the desired transposed matrix

    Returns:
        Matrix: the result of the transpose the matrix
    """
    result = Matrix(matrix.cols, matrix.rows)

    for i in range(matrix.rows):
      for j in range(matrix.cols):
        result.matrix[j][i] = matrix.matrix[i][j]

    return result

  @staticmethod
  def dot(first: Matrix, other: Matrix) -> Matrix:
    """Calculate matrix multiplication

    Args:
        first (Matrix): the first matrix
        other (Matrix): the second matrix

    Raises:
        ArithmeticError: if the first matrix cols and the second matrix rows are not equals

    Returns:
        Matrix: result of the matrix multiplication
    """
    if first.cols == other.rows:
      result = Matrix(first.rows, other.cols)
      
      for i in range(result.rows):
        for j in range(result.cols):
          sum = 0
          for k in range(first.cols):
            sum += first.matrix[i][k] * other.matrix[k][j]
          
          result.matrix[i][j] = sum

      return result
    else:
        raise ArithmeticError("the two matrixes need to be the same size: {}*{} {}*{}"
        .format(len(first.matrix), len(first.matrix[0]), len(other.matrix), len(other.matrix[0])))


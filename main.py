import evp


def main22():

  matA = [
    [4, 1],
    [1, 4],
  ]

  result = evp.jacobi_method(matA, b_verbose=True, b_plot=True)
  print(result)

  input('=== Press Enter to continue ===')


def main33():

  matA = [
    [4, 2, 1],
    [2, 4, 2],
    [1, 2, 4],
  ]

  result = evp.jacobi_method(matA, b_verbose=True, b_plot=True)

  print(result)


def main44():

  matA = [
    [8, 4, 2, 1],
    [4, 8, 4, 2],
    [2, 4, 8, 4],
    [1, 2, 4, 8],
  ]

  result = evp.jacobi_method(matA, b_verbose=True, b_plot=True)

  print(result)


def main77():
  import numpy.random as nr

  nr.seed()

  matA = nr.random((7,7))
  matA = matA @ matA.T

  result = evp.jacobi_method(matA.tolist(), b_plot=True)

  print(result)


if "__main__" == __name__:
  main22()
  main33()
  main77()

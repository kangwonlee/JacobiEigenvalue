import evp


def main():

  matA = [
    [4, 2, 1],
    [2, 4, 2],
    [1, 2, 4],
  ]

  result = evp.jacobi_method(matA, b_verbose=True)

  print(result)


if "__main__" == __name__:
  main()

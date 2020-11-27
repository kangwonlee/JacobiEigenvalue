"""
고유치 문제 모듈
Eigenvalue Problem Module

python 의 list 의 list 을 이용하는 행렬로 구현함
Implement a matrix as a list of lists
"""
import math
import os

import numpy as np
import matplotlib.pyplot as plt

import matrix


def power_method(mat_a, epsilon=1e-9, b_verbose=False):
    # 행렬의 크기
    n = len(mat_a)

    # power method 초기화
    counter, lambda_k, lambda_k1, zk = initialize_power_method(n)

    while True:
        # 행렬 곱셈 후 가장 큰 성분을 탐색
        lambda_k1, yk1 = iterate_power_method(mat_a, zk, n, lambda_k1)

        # 이전 단계의 가장 큰 요소와 비교
        if abs(lambda_k1 - lambda_k) < epsilon:
            break
        lambda_k = lambda_k1

        # 사용이 왼료된 y1 벡터의 메모리 공간을 반환
        del yk1
        counter += 1

    if b_verbose:
        print("power method counter = %d" % counter)

    return lambda_k1, zk


def initialize_power_method(n):
    # 가장 큰 고유치를 담게 될 변수
    lambda_k = 0.0
    lambda_k1 = 1.0
    # 위 고유치의 고유 벡터를 저장할 장소
    zk = [1.0] * n
    counter = 0
    # k : 반복횟수
    # i : i 번째 고유치, 고유 벡터
    return counter, lambda_k, lambda_k1, zk


def iterate_power_method(mat_a, zk, n, lambda_k1):
    # 행렬 곱셈
    # k 가 큰 값이라면 z_k 는 첫번째 고유벡터와 거의 같은 방향이므로
    # y_k+1 = mat_a z_k = lambda_1 z_k
    # z_k 의 가장 큰 요소는 1 이었으므로
    # y_k+1 의 가장 큰 요소가 lambda_1 인 것이라고 볼 수 있다.
    yk1 = matrix.mul_mat_vec(mat_a, zk)
    # yk1 벡터에서 절대값이 가장 큰 요소를 찾음
    lambda_k1 = abs(yk1[0])
    for yk1_i in yk1[1:]:
        if abs(yk1_i) > abs(lambda_k1):
            lambda_k1 = yk1_i

    # 위에서 찾은 값으로 yk1 모든 요소를 나누어서 zk 벡터에 저장
    # "위에서 찾은 값으로 yk1 을 normalize 한다"
    # zk 의 가장 큰 요소는 1이 됨
    for i in range(n):
        zk[i] = yk1[i] / lambda_k1

    return lambda_k1, yk1


def search_max_off_diagonal(mat_a0, n):
    r = 0
    s = 1
    ars = mat_a0[r][s]
    abs_ars = abs(ars)

    for i in range(n - 1):
        for j in range(i + 1, n):
            aij = abs(mat_a0[i][j])
            if aij > abs_ars:
                r = i
                s = j
                abs_ars = aij
                ars = mat_a0[i][j]

    return abs_ars, ars, r, s


def calc_theta(ars, arr, ass):
    theta_rad = 0.5 * math.atan2((2.0 * ars), (arr - ass))
    return theta_rad


def jacobi_method(mat_a, epsilon=1e-9, b_verbose=False):
    mat_a0, mat_x, n, counter = initialize_jacobi_method(mat_a)

    #########################
    while True:
        abs_ars, ars, r, s = search_max_off_diagonal(mat_a0, n)

        if abs_ars < epsilon:
            break
        if b_verbose:
            print("ars = %s" % ars)
            print("r, s = (%g, %g)" % (r, s))

        arr, ass, cos, sin = get_givens_rotation_elements(ars, b_verbose, mat_a0, r, s)

        jacobi_rotation(ars, arr, ass, cos, sin, mat_a0, mat_x, n, r, s)

        counter += 1

        if b_verbose:
            print("mat_a%03d" % counter)
            matrix.show_mat(mat_a0)
            print("mat_x%03d" % counter)
            matrix.show_mat(mat_x)
            plt.matshow(
              np.hstack((
                np.array(mat_a0), np.array(mat_x)
              ))
            )
            plt.title(f"iteration{counter:03d}")
            plt.savefig(f"iteration{counter:03d}.png")


    return mat_a0, mat_x


def jacobi_rotation(ars, arr, ass, cos, sin, mat_a0, mat_x, n, r, s):
    for k in range(n):
        if k == r:
            pass
        elif k == s:
            pass
        else:
            akr = mat_a0[k][r]
            aks = mat_a0[k][s]
            mat_a0[r][k] = akr * cos + aks * sin
            mat_a0[s][k] = aks * cos - akr * sin

            mat_a0[k][r] = mat_a0[r][k]
            mat_a0[k][s] = mat_a0[s][k]

        xkr = mat_x[k][r]
        xks = mat_x[k][s]
        mat_x[k][r] = xkr * cos + xks * sin
        mat_x[k][s] = xks * cos - xkr * sin
    mat_a0[r][r] = arr * cos * cos + 2.0 * ars * sin * cos + ass * sin * sin
    mat_a0[s][s] = arr * sin * sin - 2.0 * ars * sin * cos + ass * cos * cos
    mat_a0[r][s] = mat_a0[s][r] = 0.0


def get_givens_rotation_elements(ars, b_verbose, mat_a0, r, s):
    arr = mat_a0[r][r]
    ass = mat_a0[s][s]
    theta_rad = calc_theta(ars, arr, ass)
    if b_verbose:
        print("theta = %s (deg)" % (theta_rad * 180 / math.pi))
    cos = math.cos(theta_rad)
    sin = math.sin(theta_rad)
    return arr, ass, cos, sin


def initialize_jacobi_method(mat_a):

    remove_all_figure_files()

    n = len(mat_a)
    mat_a0 = matrix.alloc_mat(n, n)
    for i in range(n):
        for j in range(n):
            mat_a0[i][j] = mat_a[i][j]
    mat_x = matrix.get_identity_matrix(n)
    counter = 0
    return mat_a0, mat_x, n, counter


def remove_all_figure_files(ext:str='png'):
  for filename in os.listdir():
    if os.path.splitext(filename)[-1].lower().endswith(ext.lower()):
      os.remove(filename)

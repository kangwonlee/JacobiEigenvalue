"""
행렬 모듈
python 의 list 의 list 을 이용하여 행렬을 구현함
"""

import random

random.seed()


def alloc_vec(n):
    return [0.0] * n


def alloc_mat(m, n):
    result = alloc_vec(m)
    for k in range(m):
        result[k] = alloc_vec(n)
    return result


def alloc_mat_bad_00(m, n):
    row = alloc_vec(n)
    result = []
    for k in range(m):
        result.append(row)
    # Q: Python 에서 이렇게 행렬을 생성할 경우 안전한가?
    return result


def alloc_mat_bad_01(m, n):
    result = [[0.0] * n] * m
    # Q: Python 에서 이렇게 행렬을 생성할 경우 안전한가?
    return result


def shape(mat_a):
    # Q: python 의 list 의 특성상 행렬의 크기를 이렇게 확인하는 것은 안전한가?
    return len(mat_a), len(mat_a[0])


def mat_like(mat_a):
    m_row, n_col = shape(mat_a)
    mat_c = alloc_mat(m_row, n_col)
    return mat_c, m_row, n_col


def scalar_mul_mat(a, mat_a):
    mat_c, m_row, n_col = mat_like(mat_a)
    for i in range(m_row):
        for j in range(n_col):
            mat_c[i][j] = a * mat_a[i][j]
    return mat_c


def add_mat(mat_a, mat_b):
    # Q: 연산 전 어떤 확인 과정이 있으면 좋을 것인가?
    mat_c, m_row, n_col = mat_like(mat_a)
    for i in range(m_row):
        for j in range(n_col):
            mat_c[i][j] = mat_a[i][j] + mat_b[i][j]
    return mat_c


def transpose_mat(mat_a):
    m_row, n_col = shape(mat_a)
    mat_c = alloc_mat(n_col, m_row)
    for i in range(m_row):
        for j in range(n_col):
            mat_c[j][i] = mat_a[i][j]
    # Q: 다음 구문으로 transpose 된 행렬을 구할 수 있는가?
    #    mat_c = zip(*mat_a)
    return mat_c


def mul_mat_vec(mat_a, x):
    m_row, n_col = shape(mat_a)
    c = alloc_vec(m_row)
    for i in range(m_row):
        c[i] = 0.0
        for k in range(n_col):
            # Multiply and ACcumulation MAC Operation
            c[i] += mat_a[i][k] * x[k]
    return c


def mul_mat(mat_a, mat_b):
    m_row_a, n_col_a = shape(mat_a)
    m_row_b, n_col_b = shape(mat_b)
    mat_c = alloc_mat(m_row_a, n_col_b)
    for i in range(m_row_a):
        for j in range(n_col_b):
            mat_c[i][j] = 0.0
            for k in range(n_col_a):
                # Multiply and ACcumulation MAC Operation
                mat_c[i][j] += mat_a[i][k] * mat_b[k][j]
    return mat_c


def get_cofactor_matrix(mat_a, i, j):
    m_row_a, n_col_a = shape(mat_a)
    mat_c = alloc_mat(m_row_a - 1, n_col_a - 1)
    i_row_c = 0
    j_col_c = 0
    for i_row_a in range(m_row_a):
        if i_row_a != i:
            for j_col_a in range(n_col_a):
                if j_col_a != j:
                    # print("(%d,%d)<-(%d,%d)"%₩
                    #    (i_row_c,j_col_c,i_row_a,j_col_a))
                    mat_c[i_row_c][j_col_c] = mat_a[i_row_a][j_col_a]
                    j_col_c += 1
            j_col_c = 0
            i_row_c += 1
    return mat_c


def det(mat_a):
    m_row, n_col = shape(mat_a)
    if m_row == 2:
        result = mat_a[0][0] * mat_a[1][1] - mat_a[1][0] * mat_a[0][1]
    else:
        result = 0
        for j_col in range(n_col):
            mat_cofactor = get_cofactor_matrix(mat_a, 0, j_col)
            result += det(mat_cofactor) * ((-1) ** j_col)
    return result


def adjugate_matrix(mat_a):
    mat_c, m_row, n_col = mat_like(mat_a)
    for i in range(m_row):
        for j in range(n_col):
            mij = det(get_cofactor_matrix(mat_a, i, j))
            mat_c[i][j] = mij * ((-1) ** (i + j))
    return transpose_mat(mat_c)


def row_mul_scalar(mat_a, i_row, a):
    n = len(mat_a[i_row])
    for k in range(n):
        mat_a[i_row][k] *= a


def row_mul_add(mat_a, i_row, j_row, a, k_from=0):
    n = len(mat_a[i_row])
    for k in range(k_from, n):
        mat_a[i_row][k] += a * mat_a[j_row][k]


def get_identity_matrix(m_row):
    mat_i = alloc_mat(m_row, m_row)
    for i_pivot in range(m_row):
        mat_i[i_pivot][i_pivot] = 1.0
    return mat_i


def get_random_vector(n):
    x = alloc_vec(n)
    for j in range(n):
        x[j] = random.random() - 0.5
    return x


def get_random_mat(m_row, n_col):
    mat_a = []
    for i in range(m_row):
        mat_a.append(get_random_vector(n_col))
    return mat_a


def print_mat_elem(mat, mat_name, m_row, n_col):
    temp = mat_name + '[%d][%d] = %s'
    for i_row in range(m_row):
        for j_col in range(n_col):
            print(temp % (i_row, j_col, mat[i_row][j_col]))


def show_mat(m, form='%8.3f'):
    msg = '['
    for row in m:
        msg += ' ['
        for col in row:
            msg += form % col
            msg += ','
        msg += '],\n'
    msg += ']'

    print (msg.replace('[ [', '[[').replace(',]', ']').replace('],\n]', ']]'))


def main():
    # access variable
    mat_a = [[11, 12],
             [21, 22]]
    print_mat_elem(mat_a, 'mat_a', 2, 2)

    mat_b = [[1100, 1200],
             [2100, 2200]]
    print_mat_elem(mat_b, 'mat_b', 2, 2)

    mat_c = add_mat(mat_a, mat_b)
    print_mat_elem(mat_c, 'mat_c', 2, 2)

    mat_a2 = scalar_mul_mat(2.0, mat_a)
    print_mat_elem(mat_a2, 'mat_a2', 2, 2)


if "__main__" == __name__:
    main()

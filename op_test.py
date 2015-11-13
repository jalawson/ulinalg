'''
    Test routines for __OP__ for linalg module
'''
import linalg

X = linalg.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
Ymatrix = linalg.matrix([[1,2,3]])
Zmatrix = linalg.matrix([[1,2,3,4]])

def row(X):
    return [i*X.rstride+j for i in range(X.m) for j in range(X.n)]

def col(X):
    return [i*X.rstride+j for j in range(X.n) for i in range(X.m)]

def row_t(X):
    return [i*X.cstride+j for j in range(X.m) for i in range(X.n)]

def col_t(X):
    return [i*X.cstride+j for i in range(X.n) for j in range(X.m)]

print(X)

print(row(X))
# should be:
# 0 1 2 3 4 5 6 7 8 9 10 11

print(col(X))
# should be
# 0 3 6 9 1 4 7 10 2 5 8 11

print(row_t(X.T))
# should be
# 0 3 6 9 1 4 7 10 2 5 8 11

print(col_t(X.T))
# should be:
# 0 1 2 3 4 5 6 7 8 9 10 11


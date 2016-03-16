# test cases for the microulinalg package

import sys
import umatrix
import ulinalg

# ulinalg tries to determine the machine epsilon
# operations resulting in irrational numbers seem to require a tolerance of at least 2*eps
eps = umatrix.flt_eps

def matrix_compare(X, Y, tol=0):
    # checks for equal elements and identical shapes
    return all([not(abs(X[i,j] - Y[i,j]) > tol) for j in range(X.size(2)) for i in range(X.size(1))])

def construct():

    result = {}

    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x10 = umatrix.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

    result['square test 1'] = x10.is_square
    result['square test 2'] = not x11.is_square
    result['transpose column vector'] = matrix_compare(umatrix.matrix([1,2,3,4], cstride=1, rstride=1).T, umatrix.matrix([[1,2,3,4]]))
    result['transpose test square'] = matrix_compare(x10.transpose(), umatrix.matrix([[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14],[3, 7, 11, 15]]))
    result['transpose property'] = matrix_compare(x10.T, umatrix.matrix([[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14],[3, 7, 11, 15]]))
    # check for shape change view
    x12 = x11
    result['shape'] = x11.shape == (4, 3)
    x11.shape=(3, 4)
    result['shape change'] = matrix_compare(x11, umatrix.matrix([[0,1,2,4],[5,6,8,9],[10,12,13,14]])) and (x11 == x12)
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    # check for shape change copy
    x12 = x11.reshape((3,4))
    result['shape copy'] = matrix_compare(x12, umatrix.matrix([[0,1,2,4],[5,6,8,9],[10,12,13,14]])) and (x12.shape != x11.shape)

    return result

def equality():

    result = {}

    x10 = umatrix.matrix([[0.03,1.2,2.45],[4.5,5.45,6.98],[8,9.0001,10.2],[12.123,13.45,14.0]])
    x11 = umatrix.matrix([[0.03,1.2,2.45],[4.5,5.45,6.98],[8,9.0001,10.2],[12.123,13.45,14.0]])
    x12 = umatrix.matrix([[0.03,1.2,2.451],[4.5,5.45,6.98],[9,9.0002,10.2],[12.123,13.45+eps,14.0]])
    x13 = umatrix.matrix([[0.03,1.2,2.451],[4.5,5.45,6.98],[9,9.0003,10.2],[12.123,13.450001,14.0]])

    result['x == y and x.__eq__(y)'] = (x10 == x11) and (x10.__eq__(x11))
    result['umatrix.matrix_isclose(x, y) True'] = matrix_compare(x10, x11)
    result['umatrix.matrix_isclose(x, y) False'] = matrix_compare(x10, x12) == False
    result['umatrix.matrix_isclose(x, y, tol) False tol'] = matrix_compare(x12, x13, tol=eps/2) == False
    result['umatrix.matrix_isclose(x, y, tol) True tol'] = matrix_compare(x12, x13, tol=0.001)
    try:
        result['umatrix.matrix_equal(x, y)'] = umatrix.matrix_equal(x10, x12) == False
    except Exception as e:
        result['umatrix.matrix_equal(x, y)'] = (False, e)
    result['umatrix.matrix_equiv(x, y) same shape'] = umatrix.matrix_equiv(x10, x11)
    result['umatrix.matrix_equiv(x, y.reshape) shape'] = umatrix.matrix_equiv(x10, x11.reshape((3,4)))

    return result

def element_wise():

    result = {}

    X = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10]])

    try:
        result['matrix - matrix'] = matrix_compare(X-X, umatrix.matrix([[0,0,0],[0,0,0],[0,0,0]]))
    except Exception as e:
        result['matrix + list row default'] = (False, e)
    try:
        result['matrix - matrix view'] = matrix_compare(X-X.T, umatrix.matrix([[0,-3,-6],[3,0,-3],[6,3,0]]))
    except Exception as e:
        result['matrix + list row default'] = (False, e)
    try:
        result['matrix view - matrix'] = matrix_compare(X.T-X, umatrix.matrix([[0,-3,-6],[3,0,-3],[6,3,0]]).T)
    except Exception as e:
        result['matrix + list row default'] = (False, e)

    return result

def list_ops():

    result = {}

    X = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    Ylist = [1,2,3]
    Zlist = [1,2,3,4]
    Ymatrix = umatrix.matrix([[1,2,3]])
    Zmatrix = umatrix.matrix([[1,2,3,4]])

    try:
        result['matrix + list row defualt'] = matrix_compare(X+Ylist, umatrix.matrix([[1,3,5],[5,7,9],[9,11,13],[13,15,17]]))
    except Exception as e:
        result['matrix + list row default'] = (False, e)
    try:
        result['matrix + list row default broadcast error'] = matrix_compare(X+Zlist, umatrix.matrix([[1,3,5],[5,7,9],[9,11,13],[13,15,17]]))
    except Exception as e:
        result['matrix + list row default broadcast error'] = True
    try:
        result['matrix col + list'] = matrix_compare(X[:,1]+Zlist, umatrix.matrix([[2,7,12,17]]).T)
    except Exception as e:
        result['matrix col + list'] = (False, e)
    try:
        result['matrix col + list broadcast error'] = matrix_compare(X[:,1]+Ylist, umatrix.matrix([[2,7,12,17]]).T)
    except Exception as e:
        result['matrix col + list broadcast error'] = True
    try:
        result['matrix + row matrix'] = matrix_compare(X+Ymatrix, umatrix.matrix([[1,3,5],[5,7,9],[9,11,13],[13,15,17]]))
    except Exception as e:
        result['matrix + row matrix'] = (False, e)
    try:
        result['matrix + col matrix'] = matrix_compare(X+Zmatrix.T, umatrix.matrix([[1,2,3],[6,7,8],[11,12,13],[16,17,18]]))
    except Exception as e:
        result['matrix + col matrix'] = (False, e)
    try:
        result['matrix + col broadcast error'] = matrix_compare(X+Ymatrix.T, umatrix.matrix([[1,3,5],[5,7,9],[9,11,13],[13,15,17]]))
    except Exception as e:
        result['matrix + col broadcast error'] = True

    return result

def scaler():

    result = {}

    x10 = umatrix.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    try:
        result['scaler * matrix'] = matrix_compare(2*x10, umatrix.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
    except TypeError:
        result['scaler * matrix'] = (False, 'TypeError')
    try:
        result['scaler + matrix'] = matrix_compare(2.1+x10, umatrix.matrix([[2.1 , 3.1 , 4.1 , 5.1 ],[6.1 , 7.1 , 8.1 , 9.1 ],[10.1, 11.1, 12.1, 13.1],[14.1, 15.1, 16.1, 17.1]]))
    except TypeError:
        result['scaler + matrix'] = (False, 'TypeError')
    try:
        result['scaler - matrix'] = matrix_compare(1-x10, umatrix.matrix([[1, 0, -1 , -2],[-3, -4, -5, -6],[-7, -8, -9, -10],[-11, -12, -13, -14]]))
    except TypeError:
        result['scaler - matrix'] = (False, 'TypeError')
    result['matrix + scaler'] = matrix_compare(x10+2.1, umatrix.matrix([[2.1 , 3.1 , 4.1 , 5.1 ],[6.1 , 7.1 , 8.1 , 9.1 ],[10.1, 11.1, 12.1, 13.1],[14.1, 15.1, 16.1, 17.1]]))
    result['matrix - scaler'] = matrix_compare(x10-1.4, umatrix.matrix([[-1.4, -0.3999999999999999, 0.6000000000000001 , 1.6],[2.6, 3.6, 4.6, 5.6],[6.6, 7.6, 8.6, 9.6],[10.6, 11.6, 12.6, 13.6]]), tol=eps)
    result['matrix * scaler'] = matrix_compare(x10*3, umatrix.matrix([[0, 3, 6, 9],[12, 15, 18, 21],[24, 27, 30, 33],[36, 39, 42, 45]]))
    result['matrix / scaler'] = matrix_compare(x10/3, umatrix.matrix([[0.0, 0.3333333333333333, 0.6666666666666667, 1.0],
                                                                      [1.3333333333333333 , 1.6666666666666667 , 2.0, 2.3333333333333333 ],
                                                                      [2.66666666666666667 , 3.0, 3.3333333333333333 , 3.6666666666666667 ],
                                                                      [4.0, 4.3333333333333333 , 4.6666666666666667 , 5.0]]), tol=2*eps)
    result['matrix // scaler'] = matrix_compare(x10//3, umatrix.matrix([[0, 0, 0, 1],[1, 1, 2, 2],[2, 3, 3, 3],[4, 4, 4, 5]]))
    result['negate matrix'] = matrix_compare(-x10, umatrix.matrix([[0,  -1,  -2,  -3],[-4,  -5,  -6,  -7],[-8,  -9, -10, -11],[-12, -13, -14, -15]]))
    try:
        result['matrix ** scaler'] = matrix_compare(x10**2, umatrix.matrix([[ 0, 1, 4, 9],[16 , 25, 36, 49],[64, 81, 100, 121],[144, 169, 196, 225]]))
    except:
        result['matrix ** scaler'] = (False, 'NotImplemented')
    return result

def assignment():

    result = {}
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x11[1,1] = 20
    result['matrix element <- value'] = matrix_compare(x11, umatrix.matrix([[0,1,2],[4,20,6],[8,9,10],[12,13,14]]))
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x11[1,1] = [21, 23]
    result['matrix element <- list'] = matrix_compare(x11, umatrix.matrix([[0,1,2],[4,21,6],[8,9,10],[12,13,14]]))
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x11[1,:] = 21
    result['matrix row <- value'] = matrix_compare(x11, umatrix.matrix([[0,1,2],[21,21,21],[8,9,10],[12,13,14]]))
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x11[1,:] = [20, 21, 22]
    result['matrix row <- list'] = matrix_compare(x11, umatrix.matrix([[0,1,2],[20,21,22],[8,9,10],[12,13,14]]))
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x11[:,1] = 20
    result['matrix col <- value'] = matrix_compare(x11, umatrix.matrix([[0,20,2],[4,20,6],[8,20,10],[12,20,14]]))
    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x11[:,1] = [20, 21, 22, 23]
    result['matrix col <- list'] = matrix_compare(x11, umatrix.matrix([[0,20,2],[4,21,6],[8,22,10],[12,23,14]]))
    x10 = umatrix.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    x10[1:3,1:3] = 20
    result['submatrix  <- value'] = matrix_compare(x10, umatrix.matrix([[0,1,2,3],[4,20,20,7],[8,20,20,11],[12,13,14,15]]))
    x10 = umatrix.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    x10[1:3,1:3] = [20,21,22,23]
    result['submatrix  <- list'] = matrix_compare(x10, umatrix.matrix([[0,1,2,3],[4,20,21,7],[8,22,23,11],[12,13,14,15]]))
    x10 = umatrix.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    x10[1:3,1:3] = umatrix.matrix([[20,21],[22,23]])
    result['submatrix  <- matrix'] = matrix_compare(x10, umatrix.matrix([[0,1,2,3],[4,20,21,7],[8,22,23,11],[12,13,14,15]]))

    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    try:
        x11[1] = [18,19]
        result['matrix non-splice <- matrix/vector'] = not matrix_compare(x11, umatrix.matrix([[0,1,2],[18,19,6],[8,9,10],[12,13,14]]))
    except NotImplementedError:
        result['matrix non-splice <- matrix/vector'] = True
    return result

def slicing():

    result = {}

    x10 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    result['extract single element'] = x10[1,2] == 6
    result['extract a row'] = matrix_compare(x10[1,:], umatrix.matrix([[4, 5, 6]]))
    result['extract a col'] = matrix_compare(x10[:,1], umatrix.matrix([1, 5, 9, 13], cstride=1, rstride=1))
    result['extract rows'] = matrix_compare(x10[1:4,:], umatrix.matrix([[4, 5, 6],[8, 9, 10],[12, 13, 14]]))
    result['extract columns'] = matrix_compare(x10[:,1:3], umatrix.matrix([[1, 2],[5, 6],[9, 10],[13, 14]]))
    result['extract sub'] = matrix_compare(x10[1:3,1:3], umatrix.matrix([[5, 6],[9, 10]]))
    result['extract row from transpose'] = matrix_compare(x10.T[0], umatrix.matrix([[0, 4, 8, 12]]))
    result['extract col from transpose'] = matrix_compare(x10.T[:,1], umatrix.matrix([4, 5, 6], cstride=1, rstride=1))
    result['extract sub from transpose'] = matrix_compare(x10.T[1:3,1:3], umatrix.matrix([[5, 9],[6, 10]]))

    return result

def products():

    result = {}

    x11 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x10 = umatrix.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    x1 = umatrix.matrix([[0.71,-0.71,0.7],[0.71,0.71,0.5],[0,0,1]])
    y_row = umatrix.matrix([[1,0,1]])
    y_col = umatrix.matrix([1, 0, 1], cstride=1, rstride=1)
    result['matrix * scaler'] = matrix_compare(x10*2, umatrix.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
    result['matrix * matrix elementwise'] = matrix_compare(x11*x11, umatrix.matrix([[0, 1, 4],[16, 25, 36],[64, 81, 100],[144, 169, 196]]))
    result['row dot matrix 1x3 . 3x3'] = matrix_compare(ulinalg.dot(y_row,x1), umatrix.matrix([[ 0.71, -0.71,  1.7 ]]))
    try:
        result['matrix dot col 3x3 . 3x1'] = matrix_compare(ulinalg.dot(x1,y_col), umatrix.matrix([1.41, 1.21,  1.0], cstride=1, rstride=1), tol=eps)
    except ValueError:
        result['matrix dot col 3x3 . 3x1'] = False
    x = umatrix.matrix([[ 3., -2., -2.]])
    y = umatrix.matrix([[-1.,  0.,  5.]])
    x1 = umatrix.matrix([[ 3., -2.]])
    y1 = umatrix.matrix([[-1.,  0.]])
    result['cross product (x,y)'] = matrix_compare(ulinalg.cross(x,y), umatrix.matrix([[-10.0 , -13.0 , -2.0]]))
    result['cross product (y,x)'] = matrix_compare(ulinalg.cross(y,x), umatrix.matrix([[10.0 , 13.0 , 2.0]]))
    result['cross product 2 (x,y)'] = matrix_compare(ulinalg.cross(x1,y1), umatrix.matrix([[-2.0]]))
    x = umatrix.matrix([[ 3., -2., -2.],[-1.,  0.,  5.]])
    y = umatrix.matrix([[-1.,  0.,  5.]])
    try:
        result['cross product shape mismatch (x,y)'] = matrix_compare(ulinalg.cross(x,y), umatrix.matrix([[-10.0 , -13.0 , -2.0]]))
    except ValueError:
        result['cross product shape mismatch (x,y)'] = True
    return result

def iteration():

    result = {}

    x10 = umatrix.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    Z = [i for i in x10]
    result['iteration over matrix'] = Z == [umatrix.matrix([[0, 1, 2]]), umatrix.matrix([[4, 5, 6]]), umatrix.matrix([[8 , 9 , 10]]), umatrix.matrix([[12, 13, 14]])]
    Z = [i for i in x10[1,:]]
    result['iteration over row slice'] = Z == [4, 5, 6]
    Z = [i for i in x10[:,1]]
    result['iteration over col slice'] = Z == [1, 5, 9, 13]
    Z = [i for i in x10[1:3,1:2]]
    result['iteration over submatrix'] = Z == [5, 9]
    return result

def det_inv_test():

    result = {}

    # first test

    g = umatrix.matrix([[ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ],
                        [ 0.03125,  0.0625 ,  0.125  ,  0.25   ,  0.5    ,  1.     ],
                        [ 0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ],
                        [ 0.3125 ,  0.5    ,  0.75   ,  1.     ,  1.     ,  0.     ],
                        [ 0.     ,  0.     ,  0.     ,  2.     ,  0.     ,  0.     ],
                        [ 2.5    ,  3.     ,  3.     ,  2.     ,  0.     ,  0.     ]])
    g_inv = umatrix.matrix([[-192. ,  192. ,  -48. ,  -48. ,   -4. ,    4. ],
                        [ 240. , -240. ,   64. ,   56. ,    6. ,   -4. ],
                        [ -80. ,   80. ,  -24. ,  -16. ,   -3. ,    1. ],
                        [   0. ,    0. ,    0. ,    0. ,    0.5,    0. ],
                        [   0. ,    0. ,    1. ,    0. ,    0. ,    0. ],
                        [   1. ,    0. ,    0. ,    0. ,    0. ,    0. ]])
    (det,inv) = ulinalg.det_inv(g)
    result['determinant 1'] = (abs(det - 0.0078125) < 0.000001)
    result['inverse 1'] = umatrix.matrix_equal(inv, g_inv, 0.000001)

    # second test
    x = umatrix.matrix([[3.,2.,0.,1.],[4.,0.,1.,2.],[3.,0.,2.,1.],[9.,2.,3.,1.]])
    (det,inv) = ulinalg.det_inv(x)
    det_res = det == 24.0
    f = umatrix.matrix([[-0.25               , 0.25                , -0.5                , 0.25                ],
                        [0.66666666666666667 , -0.49999999999999999, 0.50000000000000001 , -0.16666666666666667],
                        [0.16666666666666667 , -0.49999999999999999, 1.00000000000000002 , -0.16666666666666667],
                        [0.41666666666666667 , 0.25                , 0.50000000000000001 , -0.41666666666666667]])
    result['inverse 2'] = matrix_compare(inv, f, tol=eps*2)#, tol=0.000000000000001)
    f[3,3] = -0.416668
    result['determinant 2'] = det_res
    result['matrix_equal True'] = umatrix.matrix_equal(inv, f, tol=0.0001)
    result['matrix_equal False'] = umatrix.matrix_equal(inv, f) == False
    x1 = umatrix.matrix([[0.71,-0.71,0.7],[0.71,0.71,0.5],[0,0,1]])
    z = ulinalg.pinv(x1)
    result['psuedo inverse'] = matrix_compare(z, umatrix.matrix([[0.7042253521126759   , 0.704225352112676    , -0.8450704225352109  ],
                                                                 [-0.704225352112676   , 0.704225352112676    , 0.1408450704225352   ],
                                                                 [1.110223024625157e-16, 5.551115123125783e-17, 0.9999999999999998   ]]), tol=2*eps)

    return result

final_results = {}
for t in [construct,
          equality,
          scaler,
          det_inv_test,
          products,
          assignment,
          slicing,
          iteration,
          list_ops,
          element_wise
         ]:
    results = t()
    print('---', t.__name__, '-'*(60-len(t.__name__)))
    for k,v in results.items():
        if type(v) == tuple:
            print('Test : {0:44s} ===> {1} : {2}'.format(k, ['    Fail','Pass'][v[0]], v[1]))
        else:
            print('Test : {0:44s} ===> {1}'.format(k, ['    Fail','Pass'][v]))
    final_results.update(results)

tests_total = len(final_results)
tests_passed = 0
for i in final_results.values():
    if type(i) == tuple:
        tests_passed += i[0]
    else:
        tests_passed += i
print('-'*60)
print('Total ==> {0:3d} Passed ==> {1:3d} Failed ==> {2:3d} Tol ==> {3}'.format(tests_total, tests_passed, tests_total-tests_passed, eps))

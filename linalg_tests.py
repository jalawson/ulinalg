# test cases for the microlinalg package

import linalg

eps = 1.0e-15

def matrix_compare(X, Y):
    return all([(abs(X[i,j] - Y[i,j]) < eps) for j in range(X.size(2)) for i in range(X.size(1))])

def s():

    result = {}

    x11 = linalg.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x10 = linalg.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    x1 = linalg.matrix([[0.71,-0.71,0.7],[0.71,0.71,0.5],[0,0,1]])
    y_row = linalg.matrix([[1,0,1]])
    y_col = linalg.matrix([1, 0, 1], cstride=1, rstride=1)

    result['square test 1'] = x10.is_square
    result['square test 2'] = not x11.is_square
    result['transpose test square'] = matrix_compare(x10.transpose(), linalg.matrix([[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14],[3, 7, 11, 15]]))
    result['transpose property'] = matrix_compare(x10.T, linalg.matrix([[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14],[3, 7, 11, 15]]))
    result['extract single element'] = x10[1,2] == 6
    result['extract a row'] = matrix_compare(x10[1,:], linalg.matrix([[4, 5, 6, 7]]))
    result['extract a col'] = matrix_compare(x10[:,1], linalg.matrix([1, 5, 9, 13], cstride=1, rstride=1))
    result['extract rows'] = matrix_compare(x10[1:4,:], linalg.matrix([[4, 5, 6, 7],[8, 9, 10, 11],[12, 13, 14, 15]]))
    result['extract columns'] = matrix_compare(x10[:,1:4], linalg.matrix([[1, 2, 3],[5, 6, 7],[9, 10, 11],[13, 14, 15]]))
    # check for shape change view
    x12 = x11
    result['shape'] = x11.shape == (4, 3)
    x11.shape=(3, 4)
    result['shape change'] = matrix_compare(x11, linalg.matrix([[0,1,2,4],[5,6,8,9],[10,12,13,14]])) and (x11 == x12)
    x11 = linalg.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    # check for shape change copy
    x12 = x11.reshape((3,4))
    result['shape copy'] = matrix_compare(x12, linalg.matrix([[0,1,2,4],[5,6,8,9],[10,12,13,14]])) and not (x11 == x12)
    x11 = linalg.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    try:
        result['scaler matrix multiplication'] = matrix_compare(2*x10, linalg.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
    except TypeError:
        result['scaler matrix multiplication'] = False
    try:
        result['scaler + matrix'] = matrix_compare(2.1+x10, linalg.matrix([[2.1 , 3.1 , 4.1 , 5.1 ],[6.1 , 7.1 , 8.1 , 9.1 ],[10.1, 11.1, 12.1, 13.1],[14.1, 15.1, 16.1, 17.1]]))
    except TypeError:
        result['scaler + matrix'] = False
    try:
        result['scaler - matrix'] = matrix_compare(1.4-x10, linalg.matrix([[1.4, 0.3999999999999999, -0.6000000000000001 , -1.6],[-2.6, -3.6, -4.6, -5.6],[-6.6, -7.6, -8.6, -9.6],[-10.6, -11.6, -12.6, -13.6]]))
    except TypeError:
        result['scaler - matrix'] = False
    result['matrix + scaler'] = matrix_compare(x10+2.1, linalg.matrix([[2.1 , 3.1 , 4.1 , 5.1 ],[6.1 , 7.1 , 8.1 , 9.1 ],[10.1, 11.1, 12.1, 13.1],[14.1, 15.1, 16.1, 17.1]]))
    result['matrix - scaler'] = matrix_compare(x10-1.4, linalg.matrix([[-1.4, -0.3999999999999999, 0.6000000000000001 , 1.6],[2.6, 3.6, 4.6, 5.6],[6.6, 7.6, 8.6, 9.6],[10.6, 11.6, 12.6, 13.6]]))
    result['matrix * scaler'] = matrix_compare(x10*2, linalg.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
    result['matrix * matrix elementwise'] = matrix_compare(x11*x11, linalg.matrix([[0, 1, 4],[16, 25, 36],[64, 81, 100],[144, 169, 196]]))
    result['row dot matrix 1x3 . 3x3'] = matrix_compare(linalg.dot(y_row,x1), linalg.matrix([[ 0.71, -0.71,  1.7 ]]))
    try:
        result['matrix dot col 3x3 . 3x1'] = matrix_compare(linalg.dot(x1,y_col), linalg.matrix([1.41, 1.21,  1.0], cstride=1, rstride=1))
    except ValueError:
        result['matrix dot col 3x3 . 3x1'] = False
    result['psuedo inverse'] = matrix_compare(linalg.pinv(x1), linalg.matrix([[0.7042253521126759   , 0.704225352112676    , -0.8450704225352109  ],
                                                                              [-0.704225352112676   , 0.704225352112676    , 0.1408450704225352   ],
                                                                              [1.110223024625157e-16, 5.551115123125783e-17, 0.9999999999999998   ]]))
    return result

def products():

    result = {}

    x = linalg.matrix([[ 3., -2., -2.]])
    y = linalg.matrix([[-1.,  0.,  5.]])
    x1 = linalg.matrix([[ 3., -2.]])
    y1 = linalg.matrix([[-1.,  0.]])
    result['cross product (x,y)'] = matrix_compare(linalg.cross(x,y), linalg.matrix([[-10.0 , -13.0 , -2.0]]))
    result['cross product (y,x)'] = matrix_compare(linalg.cross(y,x), linalg.matrix([[10.0 , 13.0 , 2.0]]))
    result['cross product 2 (x,y)'] = matrix_compare(linalg.cross(x1,y1), linalg.matrix([[-2.0]]))
    x = linalg.matrix([[ 3., -2., -2.],[-1.,  0.,  5.]])
    y = linalg.matrix([[-1.,  0.,  5.]])
    try:
        result['cross product shape mismatch (x,y)'] = matrix_compare(linalg.cross(x,y), linalg.matrix([[-10.0 , -13.0 , -2.0]]))
    except ValueError:
        result['cross product shape mismatch (x,y)'] = True
    return result

def det_inv_test():
    # det_inv test
    # det = 24.0
    # x^-1 = mat([[-0.25              , 0.25               , -0.5               , 0.25               ],
    #             [0.6666666666666667 , -0.4999999999999999, 0.5000000000000001 , -0.1666666666666667],
    #             [0.1666666666666667 , -0.4999999999999999, 1.0                , -0.1666666666666667],
    #             [0.4166666666666667 , 0.2500000000000001 , 0.5000000000000001 , -0.4166666666666667]])
    test_label = 'det_inv: '
    x = linalg.matrix([[3.,2.,0.,1.],[4.,0.,1.,2.],[3.,0.,2.,1.],[9.,2.,3.,1.]])
    (det,inv) = linalg.det_inv(x)
    det_res = det == 24.0
    f = linalg.matrix([[-0.25              , 0.25               , -0.5               , 0.25               ],
                 [0.6666666666666667 , -0.4999999999999999, 0.5000000000000001 , -0.1666666666666667],
                 [0.1666666666666667 , -0.4999999999999999, 1.0                , -0.1666666666666667],
                 [0.4166666666666667 , 0.2500000000000001 , 0.5000000000000001 , -0.4166666666666667]])
    inv_res = matrix_compare(inv, f)
    return {'determinant' : det_res, 'inverse' : inv_res}

final_results = s()
results = det_inv_test()
final_results.update(results)
results = products()
final_results.update(results)
tests_total = len(final_results)
tests_passed = sum(final_results.values())
for k,v in final_results.items():
    print('Test : {0:30s} ===> {1}'.format(k, v))
print('----------------------------------------------')
print('Total : {0:3d} : Passed {1:3d} Failed {2:3d}'.format(tests_total, tests_passed, tests_total-tests_passed))

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
    except:
        result['scaler matrix multiplication'] = False
    result['matrix * scaler'] = matrix_compare(x10*2, linalg.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
    result['matrix * matrix elementwise'] = matrix_compare(x11*x11, linalg.matrix([[0, 1, 4],[16, 25, 36],[64, 81, 100],[144, 169, 196]]))
    result['row dot matrix 1x3 . 3x3'] = matrix_compare(linalg.dot(y_row,x1), linalg.matrix([[ 0.71, -0.71,  1.7 ]]))
    try:
        result['matrix dot col 3x3 . 3x1'] = matrix_compare(linalg.dot(x1,y_col), linalg.matrix([1.41, 1.21,  1.0], cstride=1, rstride=1))
    except:
        result['matrix dot col 3x3 . 3x1'] = False
    result['psuedo inverse'] = matrix_compare(linalg.pinv(x1), linalg.matrix([[0.7042253521126759   , 0.704225352112676    , -0.8450704225352109  ],
                                                                              [-0.704225352112676   , 0.704225352112676    , 0.1408450704225352   ],
                                                                              [1.110223024625157e-16, 5.551115123125783e-17, 0.9999999999999998   ]]))
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

total_tests = 0
total_pass = 0
final_results = s()
results = det_inv_test()
final_results.update(results)
for k,v in final_results.items():
    total_tests = total_tests + 1
    if v:
        total_pass = total_pass + 1
    print('Test : {0:30s} ===> {1}'.format(k, v))
print('Total : {0:3d} : PASSED {1:3d} FAILED {2:3d}'.format(total_tests, total_pass, total_tests-total_pass))

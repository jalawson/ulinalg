# test cases for the microlinalg package

import linalg

eps = 1.0e-15

def matrix_compare(X, Y):
    return all([(abs(X[i,j] - Y[i,j]) < eps) for j in range(X.size(2)) for i in range(X.size(1))])

def s():

    result = {}

    x11 = linalg.matrix([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    x10 = linalg.matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

    result['square_test_1'] = x10.is_square
    result['square_test_2'] = not x11.is_square
    result['transpose_test_square'] = matrix_compare(x10.transpose(), linalg.matrix([[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14],[3, 7, 11, 15]]))
    result['extract single element'] = x10[1,2] == 6
    result['extract a row'] = matrix_compare(x10[1,:], linalg.matrix([[4, 5, 6, 7]]))
    result['extract a col'] = matrix_compare(x10[:,1], linalg.matrix([1, 5, 9, 13], cstride=1, rstride=1))
    result['extract rows'] = matrix_compare(x10[1:4,:], linalg.matrix([[4, 5, 6, 7],[8, 9, 10, 11],[12, 13, 14, 15]]))
    result['extract columns'] = matrix_compare(x10[:,1:4], linalg.matrix([[1, 2, 3],[5, 6, 7],[9, 10, 11],[13, 14, 15]]))
    try:
        result['scaler matrix multiplication'] = matrix_compare(2*x10, linalg.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
    except:
        result['scaler matrix multiplication'] = False
    result['matrix scaler multiplication'] = matrix_compare(x10*2, linalg.matrix([[0, 2, 4, 6],[8, 10, 12, 14],[16, 18, 20, 22],[24, 26, 28, 30]]))
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

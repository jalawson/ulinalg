#microlinalg

This is a small module to provide matrix represention and operations and a few
 linear algebra matrix operations.

The idea is to be fairly compatible with Numpy arrays and similarly provided operations but not to replace all numpys functionality.

###Files:

* __linalg.py__ - matrix class and supporting linear algebra routines.
* __linalg\_tests.py__ - testing file to check most of the features.

###Currently provided features

The linalg module attempts to determine the supported types and floating point epsilon if float is supported.

The results are held in ```linalg.stypes``` and ```linalg.flt_eps``` respectively.

For example __flt\_eps__, __stypes__ under a few different platforms:
```
#    Linux          = 1E-15, [<class 'int'>, <class 'float'>, <class 'complex'>]
#    Pyboard        = 1E-6 , [<class 'int'>, <class 'float'>, <class 'complex'>]
#    WiPy           = 1    , [<class 'int'>]
```
Note: ```flt_eps``` is kind of irrelevent when all the work is done on one platform but the linalg_test file uses it to determine matrix equality.

###Matrix representation

Elements can be __int__, __float__ or __complex__ depending on the support provided by the particular
 version of MicroPython. All elements will be converted to the most _advanced_ type found (ie. if there is a __float__ in the data, all elements will be converted to __float__).
The result is held in ```linalg.dtype```.

A matrix can be constructed using a list of lists representation, where each list member is a row in the matrix.
```
X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
```
Or by supplying a list of elements and the distance to move to get to the next column (```cstride```) and row (```rstride```). 
```
X = linalg.matrix([0,1,2,3,4,5,6,7,8,9,10,11], cstride=1, rstride=3)
```

Both the above will result in:
```
X = mat([[0 ,1 ,2 ],
         [3 ,4 ,5 ],
         [6 ,7 ,8 ],
         [9 ,10,11]])
```

####Matrix slicing

```
>>> import linalg
>>> X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>> X
mat([[0 ,1 ,2 ],
     [3 ,4 ,5 ],
     [6 ,7 ,8 ],
     [9 ,10,11]])
>>>
```
```
>>> X[1,3]        # Returns a single element as __int__, __float__, __complex__
>>> 6
>>> 
```
```
>>> X[1]          # Returns a row matrix (Numpy returns a vector)
mat([[3, 4, 5]])
>>>
```
``` 
>>> X[1,:]        # Returns a row matrix (Numpy returns a vector)
mat([[3, 4, 5]])
>>>
```
```
>>> X[:,1]        # Returns a column matrix (Numpy returns a vector)
mat([[1 ],
     [4 ],
     [7 ],
     [10]])
>>>
```
```
>>> X[:,1:3]      # Returns a submatrix of every row by columns 1 and 2
mat([[1 , 2 ],
     [4 , 5 ],
     [7 , 8 ],
     [10, 11]])
>>>
```
```
>>> X[1:3,2:4]    # Returns a submatrix
mat([[5, 6],
     [8, 9]])
>>>

```


```
>>> Z=linalg.matrix([sigmoid(i) for i in range(-6,6)], cstride=1, rstride=2)
>>> Z
mat([[0.002472623156634775, 0.006692850924284857],
     [0.01798620996209156 , 0.04742587317756679 ],
     [0.1192029220221176  , 0.2689414213699951  ],
     [0.5                 , 0.7310585786300049  ],
     [0.8807970779778823  , 0.9525741268224331  ],
     [0.9820137900379085  , 0.9933071490757153  ]])
```

###Notes

####Matrix/Scaler operations
Matrices used as the LH argument of a scaler operation will work for elementwise operation.  Using a scaler as the RH argument does not work as reflected operations are not yet supported by MicroPython.

For example:

```
X+2     # matrix plus element-wise addition works
2+X     # does not work
```
(The __int__ class __\_\_add\_\___ method does not raise __NotImplementedError__ and therefore the matrix __\_\_radd\_\___ method is not invoked).

####Matrix equality
Currently X==Y will return true if X and Y have the same shape and the elements are equal. Float and complex determine equality within one flt\_eps.
Numpy does elementwise equality and provide methods such as all_close(), array_equal() to determine matrix equality.

##Classes
* matrix

##Properties
* shape     - returns the shape as a tuple (m, n).
* shape = (p, q)  - in place shape change (not a copy)
* is\_square      - returns __True__ if a square matrix
* T               - convenience property to return a transposed view

##Methods
* copy - returns a copy of the matrix
* size(axis) - returns 0-size, 1-rows, 2-columns defaults to axis=0 (to match Numpy)
* reshape(m, n) - returns a copy of the matrix with a the shape (m, n)
* transpose - returns a transpose view of the matrix

##Functions provided by linalg module
```
Z = zeros(m, n)
```
Returns a m x n matrix filled with zeros.

```
Z = ones(m, n)
```
Returns a m x n matrix filled with ones.

```
Z = eye(m)
```
Returns a m x m matrix with the diagonal of all ones.

```
(d, I) = det_inv(X)
```
> Returns the determinant and inverse of X if it exists.

> Uses row reduction to an upper triangular matrix.

Returns (0, []) if the matrix is singular.
```
Z = pinv(X)
```
Returns the results of the pseudo inverse operation of X.

Not sure how robust this is but it works for at least one example.
```
z = dot(X, Y)
```
The dot product operation.
```
Z = cross(X, Y)
```
The cross product operation.

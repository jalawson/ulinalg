#Module micro-linalg

This is a small module for MicroPython (and Python3) which provides a class for matrix represention and manipulation.

The idea is to have whatever limited functionality is implemented in this module to be compatible with 2-D Numpy arrays.

Currently supported: (see following sections and Properties, Methods and Functions below for descriptions)

* assignment
* matrix/scaler elementwise arithmatic operations
* transpose
* slicing
* iteration support
* sub-matrix assignment
* reshaping
* determinant and inverse
* pseudo inverse (may need work)
* dot product
* cross product

<hr>

###Files:

* __linalg.py__ - matrix class and supporting linear algebra routines.
* __linalg\_tests.py__ - testing file to check most of the features.
 * The three __scaler OP matrix__ operations fail under MicroPython due to reflected operations not being fully supported as yet.

<hr>

##Classes

```
matrix
```
<hr>

####Matrix instantiation

Elements can be __int__, __float__ or __complex__ depending on the support provided by the particular
 version of MicroPython. All elements will be converted to the most _advanced_ type found (ie. if there is a __float__ in the data, all elements will be converted to __float__).
The result is held in ```linalg.dtype```.

A matrix can be constructed using a list of lists representation, where each list member is a row in the matrix.
```
import linalg
X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
```
Or by supplying a list of elements and the distance (strides) to move to get to the next column (```cstride```) and row (```rstride```). 
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
<hr>

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
>>> X[1,3]        # Returns a single element as int, float, complex
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
<hr>

####Matrix assignment

Matrix elements can be assigned to using int, float, complex, list or from a matrix.
A list will assign elements in order from the source and wrap around if required.

```
>>>X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>X[1,:] = [20, 21, 22]
X
mat([[0 , 1 , 2 ],
     [20, 21, 22],
     [6 , 7 , 8 ],
     [9 , 10, 11]])
```
```
>>>X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>X[:,1] = [20, 21, 22]
X
mat([[0 , 20, 2 ],
     [3 , 21, 5 ],
     [6 , 22, 8 ],
     [9 , 20, 11]])  # wraps around to the first element of the source list
```
```
>>>X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>X[1:3,1:3] = [20, 21, 22, 23]
X
mat([[0 , 1 , 2 ],
     [3 , 20, 21],
     [6 , 22, 23],
     [9 , 10, 11]])
```
<hr>

####Iteration

Iterating over a matrix will return the rows as a list of 1xn matrices (Numpy returns a list of vectors).

```
>>>X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
>>>[i for i in X]
[mat([[0, 1, 2]]), mat([[3, 4, 5]]), mat([[6, 7, 8]]), mat([[9 , 10, 11]])]
```
Iterating over a slice of a matrix will return a list of elements.
```
>>>X = linalg.matrix([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
[i for i in X[1,:]]
[3, 4, 5]
```
<hr>

##Implementation Notes

####Matrix/Scaler operations
Matrices used as the LH argument of a scaler operation will work for elementwise operation.  Using a scaler as the RH argument does not work as reflected operations are not yet supported by MicroPython.

For example:

```
X+2     # matrix plus element-wise addition works
2+X     # does not work
```
(The __int__ class __\_\_add\_\___ method does not raise __NotImplementedError__ and therefore the ```matrix``` __\_\_radd\_\___ method is not invoked).
<hr>

####Matrix equality
Currently X==Y will return true if X and Y have the same shape and the elements are equal. Float and complex determine equality within ```flt_eps```.

Under Numpy, X==Y does elementwise equality but also provides methods such as all_close(), array_equal() to determine matrix equality.

The linalg module attempts to determine the supported types and floating point epsilon if float is supported.

The results are held in ```linalg.stypes``` and ```linalg.flt_eps``` respectively.

For example __flt\_eps__, __stypes__ under a few different platforms:
```
#    Linux          = 1E-15, [<class 'int'>, <class 'float'>, <class 'complex'>]
#    Pyboard        = 1E-6 , [<class 'int'>, <class 'float'>, <class 'complex'>]
#    WiPy           = 1    , [<class 'int'>]
```
Note: ```flt_eps``` is kind of irrelevent when all the work is done on one platform but the ```linalg_test``` file uses it to determine matrix equality.
<hr>

##Properties
```
shape
```
> Returns the shape as a tuple (m, n).

```
shape = (p, q)
```
> In place shape change (not a copy)

```
is_square
```
> Returns __True__ if a square matrix

```
T
```
> Convenience property to return a transposed view

<hr>

##Methods
```
copy
```
> Returns a copy of the matrix

```
size(axis=0)
``` 
> Returns:

> axis=0 size (n*m) for Numpy compatibilty

> axis=1 rows

> axis=2 columns

```
reshape(m, n)
``` 
> Returns a __copy__ of the matrix with a the shape (m, n)

```
transpose
```
> Returns a __view__ of the matrix transpose

<hr>

##Functions provided by linalg module
```
zeros(m, n)
```
> Returns a m x n matrix filled with zeros.

```
ones(m, n)
```
> Returns a m x n matrix filled with ones.

```
eye(m)
```
> Returns a m x m matrix with the diagonal of all ones.

```
det_inv(X)
```
> Returns the determinant and inverse of X if it exists.
> Uses Gaussian Elimination to get an upper triangular matrix.
>
> Returns (0, []) if the matrix is singular.

```
pinv(X)
```
> Returns the results of the pseudo inverse operation of X.
>
> Not sure how robust this is but it works for at least one example.

```
dot(X, Y)
```
> The dot product operation.

```
cross(X, Y)
```
> The cross product operation.

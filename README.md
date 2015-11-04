#microlinalg

This is a small module to provide matrix represention and operations and a few
 linear algebra matrix operations.

The idea is to be mostly compatible with Numpy arrays and operations but does not
even come close to providing the same number of facilities.

Currently provided features:

Matrix representation:

Elements can be int, float or complex depending on the support provided by the particular
 version of MicroPython.

A matrix can be constructed using a list of lists representation.
Each list member is a row in the matrix.
```
X = linalg.matrix([0,1,2],[3,4,5],[6,7,8],[9,10,11]])
```
Or, a list of elements and the distance to move to get to the next column and row.
```
X = linalg.matrix([0,1,2,3,4,5,6,7,8,9,10,11], cstride=1, rstride=3)
```

Both the above will result in:
```
X = mat([0 ,1 ,2 ],
        [3 ,4 ,5 ],
        [6 ,7 ,8 ],
        [9 ,10,11]])
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

##Classes:
* matrix

##Properties:
* shape
* is\_square
* T

##Methods:
* copy
* size
* reshape
* transpose

##Functions provided by linalg module:
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
Returns a m x m matrix with the diagonal all ones.

```
(d, I) = det_inv(X)
```
Returns the determinant and inverse of X if it exists.
Uses row reduction to an upper triangle matrix.
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

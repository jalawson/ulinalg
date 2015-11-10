'''

micro-linalg project to provide a small matrix / linear algebra package

The MIT License (MIT)

Copyright (c) 2015 Jamie Lawson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

stypes = [int]
flt_eps = 1


class matrix(object):

    def __init__(self, data, cstride=0, rstride=0):
        # data can be of the form
        # x = array(2) -> array(2) acts like a scaler
        # x = array([2]) -> array([2]) vector
        # x = array([1,2,3]) -> array([1,2,3]) vector length 3
        # x = array([[1,2,3]]) -> array matrix shape (1x3)
        # if cstride != 0 then interpret data as cstride and rstride
        if cstride != 0:
            if cstride == 1:
                self.n = rstride
                self.m = int(len(data) / self.n)
            else:
                self.m = cstride
                self.n = int(len(data) / self.m)
            self.cstride = cstride
            self.rstride = rstride
            self.data = data
        else:
            # else determine from list passed in
            # get the dimensions
            self.n = 1
            if type(data) == int:
                self.m = 1
            else:  # it is a list
                self.m = len(data)
                # is data[0] a list
                if (type(data[0]) == list):
                    self.n = len(data[0])
            self.data = [data[i][j]
                         for i in range(self.m) for j in range(self.n)]
            self.cstride = 1
            self.rstride = self.n
        # ensure all elements are of the same type
        ty = max([stypes.index(type(i)) for i in self.data])
        self.data = [stypes[ty](i) for i in self.data]
        self.dtype = stypes[ty]

    def __len__(self):
        return self.m

    def __eq__(self, other):
        d = all([self.data[i] == other.data[i] for i in range(self.size())])
        return d and (self.shape == other.shape)

    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __iter__(self):
        self.cur = 0
        # determine proper axis
        if self.m == 1:
            self.cnt_lim = self.n
        else:
            self.cnt_lim = self.m
        return self

    def __next__(self):
        '''
        Returns a matrix if m > 1
        else the next numeric element of the vector.
        differs from numpy (returns vectors if selected bia slice)
        '''
        if self.cur >= self.cnt_lim:
            raise StopIteration
        self.cur = self.cur + 1
        if self.m == 1:
            return self.data[self.cur-1]
        else:
            return self[self.cur-1]

    def slice_to_offset(self, r0, r1, c0, c1):
        # check values and limit them
        nd = [self.data[i*self.rstride + j*self.cstride]
              for i in range(r0, r1) for j in range(c0, c1)]
        return matrix(nd, cstride=1, rstride=(c1-c0))

    def slice_indices(self, index, axis=0):
        # this section is to get around the missing slice.indices() method
        # the following should work when uPy implements the slice.indices()
        # method
        #     midx = index[0].indices(self.m)
        #     nidx = index[1].indices(self.n)
        st = str(index).split(',')
        if st[0][6:] == 'None':
            s0 = 0
            p0 = self.shape[axis]
        else:
            s0 = int(st[0][6:])
            p0 = int(st[1][1:])
        return (s0, p0)

    def __getitem__(self, index):
        if type(index) == tuple:
            # int and int
            # int and slice
            # slice and int
            # slice and slice
            if isinstance(index[0], int):
                s0 = index[0]
                p0 = s0+1
            else:  # row slice
                s0, p0 = self.slice_indices(index[0], 0)
            if isinstance(index[1], int):
                s1 = index[1]
                p1 = s1+1
            else:  # column slice
                s1, p1 = self.slice_indices(index[1], 1)
        elif type(index) == list:
            # list of indices etc
            raise NotImplementedError('Fancy indexing')
        else:
            # type is int? This will default to returning a row
            s0 = index
            p0 = s0 + 1
            s1 = 0
            p1 = self.n
        # resultant matrix
        z = self.slice_to_offset(s0, p0, s1, p1)
        # if it's a single entry then return that entry as int, float etc.
        if (p0 == s0 + 1) and (p1 == s1 + 1):
            return z.data[0]
        else:
            return z

    def __setitem__(self, index, val):
        if type(index) != tuple:
            # need to make it a slice without the slice function
            raise NotImplementedError('Need to use the slice [1,:] format.')
        # int and int => single entry gets changed (if val in [int,float])
        # int and slice => row and columns take on val if val fits integrally
        # slice and int
        # slice and slice
        # get the slice_to_offsets and run through the submatrix assigning
        # values from val using mod on the rows and columns
        if isinstance(index[0], int):
            s0 = index[0]
            p0 = s0+1
        else:  # slice
            s0, p0 = self.slice_indices(index[0], 0)
        if isinstance(index[1], int):
            s1 = index[1]
            p1 = s1+1
        else:  # slice
            s1, p1 = self.slice_indices(index[1], 1)
        if type(val) == matrix:
            val = val.data
        elif type(val) != list:
            val = [val]
        if not all([type(i) in stypes for i in val]):
            raise ValueError('Non numeric entry')
        else:
            k = 0
            for i in range(s0, p0):
                for j in range(s1, p1):
                    self.data[i*self.rstride + j*self.cstride] = (
                              self.dtype(val[k]))
                    k = (k + 1) % len(val)

    # there is also __delitem__

    # def __str__(self):
    def __repr__(self):
        # things that use __str__ will fallback to __repr__
        # find max string field size for the matrix elements
        l = 0
        for i in self.data:
            l = max(l, len(repr(i)))
        s = 'mat(['
        r = 0
        for i in range(self.m):
            c = 0
            s = s + '['
            for j in range(self.n):
                s1 = repr(self.data[r + c])
                s = s + s1 + ' '*(l-len(s1))
                if (j < (self.n-1)):
                    s = s + ', '
                c = c + self.cstride
            if (i < (self.m-1)):
                s = s + '],\n     '
            else:
                s = s + ']'
            r = r + self.rstride
        s = s + '])'
        return s

    # Reflected operations are not yet implemented in MicroPython

    def __add__(self, a):
        # matrix + scaler elementwise scaler adition
        if type(a) in stypes:
            ndata = [self.data[i]+a for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride=self.rstride)
        elif (type(a) == matrix):
            # element by element multiplication
            ndata = [[self[i, j]+a[i, j]
                     for j in range(self.n)] for i in range(self.m)]
            return matrix(ndata)
        raise NotImplementedError()

    def __radd__(self, a):
        return self.__add__(a)

    def __sub__(self, a):
        # matrix - scaler elementwise scaler subtraction
        if type(a) in stypes:
            ndata = [self.data[i]-a for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride=self.rstride)
        elif (type(a) == matrix):
            # element by element subtraction
            ndata = [[self[i, j]-a[i, j]
                     for j in range(self.n)] for i in range(self.m)]
            return matrix(ndata)
        raise NotImplementedError()

    def __rsub__(self, a):
        # scaler - matrix elementwise scaler subtraction
        if type(a) in stypes:
            ndata = [a - self.data[i] for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride=self.rstride)
        raise NotImplementedError()

    def __mul__(self, a):
        # matrix * scaler element by scaler multiplication
        if type(a) in stypes:
            ndata = [self.data[i]*a for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride=self.rstride)
        elif (type(a) == matrix):
            # element by element multiplication
            ndata = [[self[i, j]*a[i, j]
                     for j in range(self.n)] for i in range(self.m)]
            return matrix(ndata)
        raise NotImplementedError()

    def __rmul__(self, a):
        ''' uPy int,float __mul__ does not yet implement the reflected
            operation call if an operation raises NotImplemented exception
            so __rmul__ never gets called.
        '''
        # scaler * matrix element by element multiplication
        if type(a) in stypes:
            ndata = [self.data[i]*a for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride=self.rstride)

    def __truediv__(self, a):
        # matrix / scaler element by scaler multiplication
        if type(a) in stypes:
            try:
                ndata = [self.data[i]/a for i in range(len(self.data))]
                return matrix(ndata,
                              cstride=self.cstride, rstride=self.rstride)
            except ZeroDivisionError:
                raise ZeroDivisionError('division by zero')
        else:
            raise NotImplementedError()

    def __floordiv__(self, a):
        # matrix / scaler element by scaler multiplication
        if type(a) in stypes:
            try:
                ndata = [self.data[i]//a for i in range(len(self.data))]
                return matrix(ndata,
                              cstride=self.cstride, rstride=self.rstride)
            except ZeroDivisionError:
                raise ZeroDivisionError('division by zero')
        else:
            raise NotImplementedError()

    def copy(self):
        """ Return a copy of matrix, not just a view """
        return matrix([i for i in self.data],
                      cstride=self.cstride, rstride=self.rstride)

    def size(self, axis=0):
        """ 0 entries
            1 rows
            2 columns
        """
        return [self.m*self.n, self.m, self.n][axis]

    def __len__(self):
        return self.m

    @property
    def shape(self):
        return (self.m, self.n)

    @shape.setter
    def shape(self, nshape):
        """ check for proper length """
        if (nshape[0]*nshape[1]) == self.size():
            self.m, self.n = nshape
            self.cstride = 1
            self.rstride = self.n
        else:
            raise ValueError('total size of new matrix must be unchanged')
        return self

    @property
    def is_square(self):
        return self.m == self.n

    def reshape(self, nshape):
        """ check for proper length """
        X = self.copy()
        X.shape = nshape
        return X

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        """ Return a view """
        return matrix(self.data, cstride=self.rstride, rstride=self.cstride)


# matrix operations

def zeros(m, n):
    return matrix([[0 for i in range(n)] for j in range(m)])


def ones(m, n):
    return matrix([[1 for i in range(n)] for j in range(m)])


def eye(m):
    Z = zeros(m, m)
    for i in range(m):
        Z[i, i] = 1
    return Z


def det_inv(x):
    ''' Return (det(x) and inv(x))

        Operates on a copy of x
        Using elementary row operations convert X to an upper matrix
        the product of the diagonal = det(X)
        Continue to convert X to the identity matrix
        All the operation carried out on the original identity matrix
        makes it the inverse of X
    '''
    if not x.is_square:
        raise ValueError('Matrix must be square')
    else:
        # divide each row element by [0] to give a one in the first position
        # (may have to find a row to switch with if first element is 0)
        x = x.copy()
        inverse = eye(len(x))*1.0
        sign = 1
        factors = []
        p = 0
        while p < len(x):
            d = x[p,  p]
            if abs(d) < flt_eps:
                # pivot == 0 need to swap a row
                # check if swap row also has a zero at the same position
                np = 1
                while (p+np) < len(x) and x[p+np,  p] < flt_eps:
                    np += 1
                if (p+np) == len(x):
                    # singular
                    return [0,  []]
                # swap rows
                z = x[p+np]
                x[p+np, :] = x[p]
                x[p, :] = z
                # do identity
                z = inverse[p+np]
                inverse[p+np, :] = inverse[p]
                inverse[p, :] = z
                # change sign of det
                sign = -sign
                continue
            factors.append(d)
            # change target row
            for n in range(p, len(x)):
                x[p, n] = x[p, n] / d
            # need to do the entire row for the inverse
            for n in range(len(x)):
                inverse[p, n] = inverse[p, n] / d
            # eliminate position in the following rows
            for i in range(p+1, len(x)):
                # multiplier is that column entry
                t = x[i, p]
                for j in range(p, len(x)):
                    x[i, j] = x[i, j] - (t * x[p, j])
                for j in range(len(x)):
                    inverse[i, j] = inverse[i, j] - (t * inverse[p, j])
            p += 1
        s = sign
        for i in factors:
            s = s * i  # determinant
        # travel through the rows eliminating upper diagonal non-zero values
        for i in range(len(x)-1):
            # final row should already be all zeros
            # except for the final position
            for p in range(i+1, len(x)):
                # multiplier is that column entry
                t = x[i, p]
                for j in range(i+1, len(x)):
                    x[i, j] = x[i, j] - (t * x[p, j])
                for j in range(len(x)):
                    inverse[i, j] = inverse[i, j] - (t * inverse[p, j])
        return (s,  inverse)


def pinv(X):
    ''' Calculates the pseudo inverse Adagger = (A'A)^-1.A' '''
    Xt = X.transpose()
    d, Z = det_inv(dot(Xt, X))
    return dot(Z, Xt)


def dot(X, Y):
    ''' Dot product '''
    if X.size(2) == Y.size(1):
        Z = []
        for k in range(X.size(1)):
            for j in range(Y.size(2)):
                Z.append(sum([X[k, i] * Y[i, j] for i in range(Y.size(1))]))
        return matrix(Z,  cstride=1,  rstride=Y.size(2))
    else:
        raise ValueError('shapes not aligned')


def cross(X,  Y):
    ''' Cross product '''
    if (X.n in (2,  3)) and (Y.n in (2,  3)):
        if X.m == Y.m:
            Z = []
            for k in range(min(X.m,  Y.m)):
                z = X[k, 0]*Y[k, 1] - X[k, 1]*Y[k, 0]
                if (X.n == 3) and (Y.n == 3):
                    Z.append([X[k, 1]*Y[k, 2] - X[k, 2]*Y[k, 1],
                              X[k, 2]*Y[k, 0] - X[k, 0]*Y[k, 2],  z])
                else:
                    Z.append([z])
            return matrix(Z)
        else:
            raise ValueError('shape mismatch')
    else:
        raise ValueError('incompatible dimensions for cross product
                         (must be 2 or 3)')


def mf():
    ''' Determine floating point resolution '''
    i = 0
    x = '1.0'
    y = x
    z = float(x)
    while y == str(z):
        x = str(x) + '1'
        z = float(x)
        y = str(x)
        i = i + 1
    return i

# Determine supported types
try:
    stypes.append(float)
    flt_eps = 1./(10**mf())
except:
    pass
try:
    stypes.append(complex)
except:
    pass

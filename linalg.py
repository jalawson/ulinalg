class matrix(object):

    def __init__(self, data, cstride = 0, rstride = 0):
        # data can be of the form
        # x = array(2) -> array(2) acts like a scaler
        # x = array([2]) -> array([2]) vector
        # x = array([1,2,3]) -> array([1,2,3]) vector length 3
        # x = array([[1,2,3]]) -> array matrix shape (1x3)
        #self.mat = data
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
            else: # it is a list
                self.m = len(data)
                # is data[0] a list
                if (type(data[0]) == list):
                    self.n = len(data[0])
            self.data = [data[i][j] for i in range(self.m) for j in range(self.n)]
            self.cstride = 1
            self.rstride = self.n
        #print("cstride, rstride", self.cstride, self.rstride)
        #print("m , n", self.m, self.n)
        #print(self.data)

    def __eq__(self, other):
        d = all([self.data[i] == other.data[i]] for i in range(self.size()))
        return d and (self.shape == other.shape)

    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __iter__(self):
        self.cur = 0
        return self

    def __next__(self):
        if self.cur >= self.m:
            raise StopIteration
        self.cur = self.cur + 1
        return matrix([self.data[self.cur-1]])

    def slice_to_offset(self, r0, r1, c0, c1):
        # check values and limit them
        nd = []
        for i in range(r0, r1):
            d0 = i*self.rstride + c0*self.cstride
            d1 = d0 + (c1 - c0)
            nd.extend(self.data[d0:d1])
        return matrix(nd, cstride=1, rstride=(c1-c0))

    def slice_indices(self, index):
        # this section is to get around the missing slice.indices() method
        # the following should work when the slice.indices() method is implemented in uPy
        #     midx = index[0].indices(self.m)
        #     nidx = index[1].indices(self.n)
        st = str(index).split(',')
        #print(st[0][6:], st[1][1:])
        if st[0][6:] == 'None':
            s0 = 0
            p0 = self.m
        else:
            s0 = int(st[0][6:])
            p0 = int(st[1][1:])
        return (s0, p0)

    def __getitem__(self, index):
        #print(index, type(index))
        if type(index) == tuple:
            # int and int
            # int and slice
            # slice and int
            # slice and slice
            if isinstance(index[0], int):
                s0 = index[0]
                p0 = s0+1
            else: # slice
                s0, p0 = self.slice_indices(index[0])
            if isinstance(index[1], int):
                s1 = index[1]
                p1 = s1+1
            else: #slice
                s1, p1 = self.slice_indices(index[1])
        elif type(index) == list:
            # list of indices etc
            raise NotImplementedError('Fancy indexing') 
        else:
            # type is int? This will return a row
            s0 = index
            p0 = s0 + 1
            s1 = 0
            p1 = self.n
        #print(s0, p0, s1, p1)
        # resultant matrix
        z = self.slice_to_offset(s0, p0, s1, p1)
        # if it's a single entry then return that entry as int, float etc.
        if (p0 == s0 + 1) and (p1 == s1 + 1):
            return z.data[0]
        else:
            return z

    def make_a_slice(self, i):
        return i

    def __setitem__(self, index, val):
        #print("index",index, type(val))
        if type(index) != tuple:
            # need to make it a slice without the slice function
            index = self.make_a_slice(self[index,:])
            raise NotImplementedError('Need to use the slice [1,:] format.')
        #else:
        # int and int => single entry gets changed (if val in [int,float])
        # int and slice => row and columns take on val if val fits integrly
        # slice and int
        # slice and slice
        # get the slice_to_offsets and run through the submatrix assigning
        # values from val using mod on the rows and columns
        if (type(index[0]) == int) and (type(index[1]) == int):
            if type(val) in [int, float]:
                self.data[index[0]*self.rstride + index[1]*self.cstride] = val
            else:
                raise ValueError('setting an array element with a sequence.')
        else:
            print(type(index[0]), type(index[1]))
            if isinstance(index[0], int):
                s0 = index[0]
                p0 = s0+1
            else: # slice
                s0, p0 = self.slice_indices(index[0])
            if isinstance(index[1], int):
                s1 = index[1]
                p1 = s1+1
            else: #slice
                s1, p1 = self.slice_indices(index[1])
            for i in range(s0, p0):
                d0 = i*self.rstride + s1*self.cstride
                d1 = d0 + (p1 - s1)
                k = 0
                if type(val) == matrix:
                    val = val.data
                elif type(val) != list:
                    val = [val]
                for j in range(d0,d1):
                    self.data[j] = val[k]
                    k = (k + 1) % len(val)

    # there is also __delitem__

    #def __str__(self):
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


    def __mul__(self, a):
        # matrix * scaler element by element multiplication
        #print("MULT:",type(a))
        if type(a) in [int, float]:
            ndata = [self.data[i]*a for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride= self.rstride)
        raise NotImplementedError()

    # uPy int,float __mul__ doesn't seem to implement the NotImplemented
    # thing so __rmul__ never gets called.
    def __rmul__(self, a):
        self.__mul__(a)
        # scaler * matrix element by element multiplication
        #print("RMULT:",type(a))
        if type(a) in [int, float]:
            ndata = [self.data[i]*a for i in range(len(self.data))]
            return matrix(ndata, cstride=self.cstride, rstride= self.rstride)

    def copy(self):
        # copy the data, not just a view
        return matrix([i for i in self.data], cstride=self.cstride, rstride=self.rstride)
    
    def size(self, axis = 0):
        ''' 0 entries
            1 rows
            2 columns
        '''
        return [self.m*self.n, self.m, self.n][axis]

    @property
    def length(self):
        return self.m

    @property         
    def shape(self):
        # returns True if x is square
        return (self.m, self.n)

    @property         
    def is_square(self):
        # returns True if x is square
        return self.m == self.n

    def transpose(self):
        """ Returns a view """
        # this returns a new matrix object
        #return matrix([[self.data[j][i] for j in range(self.m)] for i in range(self.n)])
        return matrix(self.data, cstride = self.rstride, rstride = self.cstride)

# matrix version operations
def zeros(m, n):
    return matrix([[0 for i in range(n)] for j in range(m)])

def ones(m, n):
    return matrix([[1 for i in range(n)] for j in range(m)])

def eye(m):
    Z = zeros(m, m)
    for i in range(m):
        Z[i,i] = 1
    return Z

def det_inv(x):
    ''' Returns (det(x) and inv(x))

        Operates on a copy of x
        Using elementary row operations convert X to an upper matrix
        the product of the diagonal = det(X)
        Continue to convert X to the identity matrix
        All the operation carried out on the original identity matrix
        makes it the inverse of X
    '''
    assert x.is_square, 'must be a square matrix'
    # divide each row element by [0] to give a one in the first position
    # (may have to find a row to switch with if first element is 0)
    x = x.copy()
    inverse = eye(x.length)
    sign = 1
    factors = []
    p = 0
    while p < x.length:
        d = x[p, p]
        if abs(d) < 1e-30:
            # pivot == 0 need to swap a row
            # need to check if swap row also has a zero at the same position
            np = 1
            while (p+np) < x.length and x[p+np, p] < 1e-30:
                np = np + 1
            if (p+np) == x.length:
                # singular
                return [0, []]
            # swap rows
            z = x[p+np]
            x[p+np,:] = x[p]
            x[p,:] = z
            # do identity
            z = inverse[p+np]
            inverse[p+np,:] = inverse[p]
            inverse[p,:] = z
            # change sign of det
            sign = -sign
            continue
        factors.append(d)
        # change target row
        for n in range(p,x.length):
            x[p,n] = x[p,n] / d
        # need to do the entire row for the inverse
        for n in range(x.length):
            inverse[p,n] = inverse[p,n] / d
        # eliminate position in the following rows
        for i in range(p+1,x.length):
            # multiplier is that column entry
            t = x[i,p]
            for j in range(p,x.length):
                x[i,j] = x[i,j] - (t * x[p,j])
            for j in range(x.length):
                inverse[i,j] = inverse[i,j] - (t * inverse[p,j])
        p = p + 1
    s = sign
    for i in factors:
        s = s * i # determinant
    # travel through the rows eliminating upper diagonal non-zero values
    for i in range(x.length-1):
        # final row should already be all zeros except for the final position
        for p in range(i+1,x.length):
            # multiplier is that column entry
            t = x[i,p]
            for j in range(i+1,x.length):
                x[i,j] = x[i,j] - (t * x[p,j])
            for j in range(x.length):
                inverse[i,j] = inverse[i,j] - (t * inverse[p,j])
    return (s, inverse)

def dot(X,Y):
    # assume X is a row vector for now
    assert X.size(2) == Y.size(1), 'shapes not aligned'
    Z = []
    for k in range(X.size(1)):
        for i in range(X.size(2)):
            s = 0
            for j in range(Y.size(2)):
                s = s + (X[k,j] * Y[j,i])
            Z.append(s)
    return matrix(Z, cstride=1, rstride=3)

def s():

    x10 = matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    # det_inv test
    # det = 24.0
    # x^-1 = mat([[-0.25              , 0.25               , -0.5               , 0.25               ],
    #             [0.6666666666666667 , -0.4999999999999999, 0.5000000000000001 , -0.1666666666666667],
    #             [0.1666666666666667 , -0.4999999999999999, 1.0                , -0.1666666666666667],
    #             [0.4166666666666667 , 0.2500000000000001 , 0.5000000000000001 , -0.4166666666666667]])
    x = matrix([[3.,2.,0.,1.],[4.,0.,1.,2.],[3.,0.,2.,1.],[9.,2.,3.,1.]])
    

    print("as a matrix")
    print(x10)
    print("Is square")
    print(x10.is_square)
    print("transpose")
    print(x10.transpose())

    #print("extract an element x10[1,2]")
    #print(x10[1,2])
    #print("extract a row x10[1,:]")
    #print(x10[1,:])
    #print("extract a column x10[:,1]")
    #print(x10[:,1])
    #print("extract some columns x10[:,1:4]")
    #print(x10[:,1:4])
    #print(x10.mat[:][1:3])

    #mx = matrix(6,1,[1.2,2,3,5,0,3])
    #my = matrix(6,1,[4,5,6,3,2,6.5])

    #print(mx)
    #print(my)
    #print(mdot(mx,my.transpose()))

    #mx1 = matrix(2,6,[1.2,2,3,5,0,3,4,5,6,3,2,6.5])
    #my1 = matrix(2,6,[4,5,6,3,2,6.5,1.2,2,3,5,0,3])
    #print(mmatxmat(mx1,my1))
    print('x10*2')
    print(x10*2)
    print('2*x10')
    print(2*x10)

def det_inv_test():
    # det_inv test
    # det = 24.0
    # x^-1 = mat([[-0.25              , 0.25               , -0.5               , 0.25               ],
    #             [0.6666666666666667 , -0.4999999999999999, 0.5000000000000001 , -0.1666666666666667],
    #             [0.1666666666666667 , -0.4999999999999999, 1.0                , -0.1666666666666667],
    #             [0.4166666666666667 , 0.2500000000000001 , 0.5000000000000001 , -0.4166666666666667]])
    x = matrix([[3.,2.,0.,1.],[4.,0.,1.,2.],[3.,0.,2.,1.],[9.,2.,3.,1.]])
    (d,i) = det_inv(x)
    #assert d == 24.0
    print(x == matrix([[3.,2.,0.,1.],[4.,0.,1.,2.],[3.,0.,2.,1.],[9.,2.,3.,1.]]))
    """
    assert is_almost_equal(i, mat([[-0.25              , 0.25               , -0.5               , 0.25               ],
                 [0.6666666666666667 , -0.4999999999999999, 0.5000000000000001 , -0.1666666666666667],
                 [0.1666666666666667 , -0.4999999999999999, 1.0                , -0.1666666666666667],
                 [0.4166666666666667 , 0.2500000000000001 , 0.5000000000000001 , -0.4166666666666667]])
                 )
    """

    y = matrix([[3.,2.,0.,1.],[4.,0.,1.,2.],[3.,0.,2.,1.],[9.,2.,3.,1.]])
    print(x == y)


def main():

    det_inv_test()

if __name__ == "__main__":
    main()

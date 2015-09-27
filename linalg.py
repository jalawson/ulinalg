class matrix():

    def __init__(self, data, cstride = 0, rstride = 0):
        print("__init__",data)
        # data can be of the form
        # x = array(2) -> array(2) acts like a scaler
        # x = array([2]) -> array([2]) vector
        # x = array([1,2,3]) -> array([1,2,3]) vector length 3
        # x = array([[1,2,3]]) -> array matrix shape (1x3)
        #self.mat = data
        # if cstride != 0 then interpret data as cstride and rstride
        print(cstride, rstride)
        if cstride != 0:
            self.n = int(len(data) / cstride)
            self.m = int(len(data) / self.n)
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
        print("cstride, rstride", self.cstride, self.rstride)
        print("m , n", self.m, self.n)
        print(self.data)

    def __iter__(self):
        self.cur = 0
        return self

    def __next__(self):
        if self.cur >= self.m:
            raise StopIteration
        self.cur = self.cur + 1
        return matrix([self.mat[self.cur-1]])

    def __getitem__(self, index):
        print(index, type(index))
        return self.data[index[0]*self.rstride + index[1]*self.cstride]

    def __oldgetitem__(self, index):
        print(index, type(index))
        is_slice=False
        if type(index) == tuple:
            # int and int
            # int and slice
            # slice and int
            # slice and slice
            # this section is to get around the missing slice.indices() method
            if isinstance(index[0], int):
                s0 = index[0]
                p0 = s0+1
            else: # slice
                is_slice = True
                s = str(index[0]).split(',')
                print(s[0][6:], s[1][1:])
                if s[0][6:] == 'None':
                    s0 = 0
                    p0 = self.m
                else:
                    s0 = int(s[0][6:])
                    p0 = int(s[1][1:])
            if isinstance(index[1], int):
                s1 = index[1]
                p1 = s1+1
            else: #slice
                is_slice = True
                s = str(index[1]).split(',')
                print(s[0][6:], s[1][1:])
                if s[0][6:] == 'None':
                    s1 = 0
                    p1 = self.n
                else:
                    s1 = int(s[0][6:])
                    p1 = int(s[1][1:])
            # this works when the slice stuff is implemented in uPy
            #midx = index[0].indices(self.m)
            #nidx = index[1].indices(self.n)
            #print("__getitem__", index, type(index[0])==slice, type(index[1])==slice)
            #print(nidx,midx)
            #return [self.mat[i][j] for j in range(*nidx) for i in range(*midx)]
        if is_slice:
            print("is_slice", s0, p0, s1, p1)
            if (p1 - s1) == 1:
                return matrix([self.data[i][s1] for i in range(s0,p0)])
            else:
                return matrix([[self.data[i][j] for j in range(s1,p1)] for i in range(s0,p0)])
        # index can be a single int
        elif isinstance(index, int):
            print("here 1")
            return matrix(self.data[index])
        else:
            return self.data[s0][s1]

    def __setitem__(self, index, val):
        print("index",index)
        self.data[index[0]*self.rstride + index[1]*self.cstride] = val

    # there is also __delitem__

    #def __str__(self):
    def __oldrepr__(self):
        # things that use __str__ will fallback to __repr__
        l = 0
        for i in self.data:
            l = max(l, len(repr(i)))
        s = 'mat({'
        for i in range(self.m):
            s = s + '['
            for j in range(self.n):
                s1 = repr(self.data[(i*self.m) + j])
                s = s + s1 + ' '*(l-len(s1))
                #s = s + repr(self.data[(i*self.m) + j])
                if (j < (self.n-1)):
                    s = s + ', '
            if (i < (self.m-1)):
                s = s + '],\n     '
            else:
                s = s + ']'
        #if type(self.data[0]) != list:
        #    s = s + repr(self.data)
        #else:
        #    for i in range(self.m):
        #        s = s + repr(self.data[i])
        #        if i < (self.m-1):
        #            s = s + ',\n   '
        s = s + '})'
        return s

    def __repr__(self):
        # things that use __str__ will fallback to __repr__
        # find max string field size for the matrix elements
        l = 0
        for i in self.data:
            l = max(l, len(repr(i)))
        s = 'mat({'
        r = 0
        for i in range(self.m):
            c = 0
            s = s + '['
            for j in range(self.n):
                s1 = repr(self.data[r + c])
                s = s + s1 + ' '*(l-len(s1))
                #s = s + repr(self.data[(i*self.m) + j])
                if (j < (self.n-1)):
                    s = s + ', '
                c = c + self.cstride
            if (i < (self.m-1)):
                s = s + '],\n     '
            else:
                s = s + ']'
            r = r + self.rstride
        s = s + '})'
        return s

    def __mul__(a,b):
        # element by element multiplication
        print("MULT:",type(a),type(b))
        if type(a) in [int, float] and type(b) in [int, float]:
            return a*b
        if type(a[0]) == int:
            for i in a:
                print(i*b)

    def copy(self):
        return self.data
    
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
        #return sum([len(i)==len(self.mat) for i in self.mat])==len(self.mat)
        return (self.m, self.n)

    @property         
    def is_square(self):
        # returns True if x is square
        #return sum([len(i)==len(self.mat) for i in self.mat])==len(self.mat)
        return self.m == self.n

    def transpose(self):
        print(self.data)
        # should try and return a view - changing an element in the transposed matrix changes the 
        # element in the original matrix
        #self.mat = [[self.mat[i][j] for i in range(len(self.mat))] for j in range(len(self.mat[0]))]
        #return [[self.mat[i][j] for i in range(len(self.mat))] for j in range(len(self.mat[0]))]
        # this returns a new matrix object
        #Z = matrix(self.n,self.m,[self.mat[i][j] for j in range(self.n) for i in range(self.m)])
        #return matrix([[self.mat[j][i] for j in range(self.m)] for i in range(self.n)])
        #return matrix(self.data, cstride = 4, rstride = 1)
        return matrix(self.data, cstride = self.rstride, rstride = self.cstride)

# matrix version operations
def mdot(x,y):
    print(x, y)
    #return sum([x[0,m]*y[m,0] for m in range(x.size(1))])
    return sum([x[0,m]*y[0,m] for m in range(x.size(1))])

def mmatxmat(x,y):
    #y = y.transpose()
    return [[ mdot(i,j) for j in y] for i in x]

def zeros(m, n):
    return matrix([[0 for i in range(n)] for j in range(m)])

def ones(m, n):
    return matrix([[1 for i in range(n)] for j in range(m)])

def eye(m):
    Z = zeros(m, m)
    for i in range(m):
        Z[i,i] = 1
    return Z

def print_matrix(x):
    for i in x:
        print(i) 

def tests():

    x10 = matrix([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

    print("as data")
    print_matrix(x10.mat)
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

def main():

    tests()

if __name__ == "__main__":
    main()

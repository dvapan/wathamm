import numpy as np

def powers(n,d):
    p = np.arange(n+1,dtype=np.float) - d
    p[p<0] = 0
    return p

def diff_coeff(n,d):
    if d == 0:
        return np.ones(n+1)
    else:
        lst = [powers(n,i) for i in range(d)]
        return np.multiply.reduce(lst)
    
def mvmonos(x, powers, diff=None):
    pass

if __name__ == "__main__":
    print(powers(3,0))
    print(powers(3,1))
    print(powers(3,2))
    print(diff_coeff(3,0))
    print(diff_coeff(3,1))
    print(diff_coeff(3,2))
    print(diff_coeff(3,3))

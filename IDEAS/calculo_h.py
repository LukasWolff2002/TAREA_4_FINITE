L = 1
r = 1/1 #Desde 0 a 1 en cualquier eje
n = 10 -1

def S1 (L, r, N):
    return (L * (1 - r)) / (1 - r**n)

def Si1 (Si, r):
    return Si*r

s1 = S1(L, r, n)
print(s1)
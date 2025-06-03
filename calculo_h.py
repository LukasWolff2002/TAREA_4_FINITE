L = 1
r = 1.3 #Desde 0 a 1 en cualquier eje
n = 10 -1

def S1 (L, r, N):
    return (L * (1 - r)) / (1 - r**n)

def Si1 (Si, r):
    return Si*r

s1 = S1(L, r, n)
s2 = Si1(s1, r)
s3 = Si1(s2, r)
s4 = Si1(s3, r)
s5 = Si1(s4, r)
s6 = Si1(s5, r)
s7 = Si1(s6, r)
s8 = Si1(s7, r)
s9 = Si1(s8, r)

print("S1:", s1)
print("S2:", s2)
print("S3:", s3)
print("S4:", s4)
print("S5:", s5)
print("S6:", s6)
print("S7:", s7)
print("S8:", s8)
print("S9:", s9)
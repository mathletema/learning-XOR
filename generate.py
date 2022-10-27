import random

set = ([0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0])
sigma = 0.008
N = 1000
M = 500

def generate():
    res = random.choice(set)
    res[0] += random.gauss(0, sigma)
    res[1] += random.gauss(0, sigma)
    return res

with open('learn.csv', 'w') as f:
    f.write('x, y, out\n')
    for i in range(N):
        tuple = generate()
        f.write(str(tuple[0]) + ', ' + str(tuple[1]) + ', ' + str(tuple[2]) + '\n')

with open('verification.csv', 'w') as f:
    f.write('x, y, out\n')
    for i in range(M):
        tuple = generate()
        f.write(str(tuple[0]) + ', ' + str(tuple[1]) + ', ' + str(tuple[2]) + '\n')
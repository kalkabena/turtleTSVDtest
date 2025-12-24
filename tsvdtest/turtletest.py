import numpy as np
import random
import turtle

np.set_printoptions(precision = 2, suppress = True, linewidth = 200, floatmode = "fixed")

pi = 3.141592653589793

def display_object(t, x, y, velx, vely):
    t.seth(0)
    t.teleport(x, y)
    t.dot()
    t.seth(np.arctan(vely / velx) * 180 / pi)
    t.forward(np.sqrt(vely * vely + velx * velx))
    
def display_line(t, x1, y1, x2, y2):
    t.teleport(x1, y1)
    t.goto(x2, y2)

def display_dot(t, x, y, color):
    color = min(color, 1)
    color = max(color, 0)
    t.teleport(x, y)
    t.color(color, color, color)
    t.dot()
    
size = 3
realsize = 4 * size
k = int(0.4142 * realsize)
normalize = True

matrix = np.zeros((realsize, realsize))
rowmeans = np.zeros(realsize)
rowstd = np.zeros(realsize)

for i in range(realsize):
    for j in range(realsize):
        if i % 4 == 0 or i % 4 == 1:
            matrix[i, j] = random.uniform(-300, 300)
        else:
            matrix[i, j] = random.uniform(-40, 40)

for i in range(realsize):
    rowmeans[i] = np.mean(matrix[i])
    rowstd[i] = np.std(matrix[i])

if normalize == True:
    matrix_norm = (matrix - rowmeans[:, None]) / (rowstd[:, None] + 1e-8)
    U, s, Vt = np.linalg.svd(matrix_norm)
else:
    U, s, Vt = np.linalg.svd(matrix)

Ur = U[:, :k]
Sr = np.diag(s[:k])
Vtr = Vt[:k, :]

approx = Ur @ Sr @ Vtr

if normalize == True:
    approx = approx * (rowstd[:, None] + 1e-8) + rowmeans[:, None]

print(f"original matrix: {matrix.size} elements")
print(f"truncated SVD matrix: {Ur.size + Sr.size + Vtr.size} elements")
print(f"smooth = {normalize}")
print(f"k = {k}")
print(s)
print(f"energy = {np.sum(s[:k]) / np.sum(s)}")
print(matrix)
print(approx)
print(np.linalg.norm(matrix - approx))

t = turtle.Turtle()

t.speed(0)
t.ht()
for i in range(4 * size * size):
    t.color("blue")
    display_line(t, matrix[i // realsize * 4, i % realsize],
    matrix[i // realsize * 4 + 1, i % realsize],
    approx[i // realsize * 4, i % realsize],
    approx[i // realsize * 4 + 1, i % realsize],)
    t.color("black")
    display_object(t, matrix[i // realsize * 4, i % realsize],
    matrix[i // realsize * 4 + 1, i % realsize],
    matrix[i // realsize * 4 + 2, i % realsize],
    matrix[i // realsize * 4 + 3, i % realsize])
    t.color("red")
    display_object(t, approx[i // realsize * 4, i % realsize],
    approx[i // realsize * 4 + 1, i % realsize],
    approx[i // realsize * 4 + 2, i % realsize],
    approx[i // realsize * 4 + 3, i % realsize])

for index, value in np.ndenumerate(matrix):
    row, col = index
    if row % 4 <= 1:
        display_dot(t, col * 4, row * 4, (value + 300) / 600)
    else:
        display_dot(t, col * 4, row * 4, (value + 40) / 80)

for index, value in np.ndenumerate(approx):
    row, col = index
    if row % 4 <= 1:
        display_dot(t, col * 4 - realsize * 4 - 40, row * 4, (value + 300) / 600)
    else:
        display_dot(t, col * 4 - realsize * 4 - 40, row * 4, (value + 40) / 80)

turtle.done()
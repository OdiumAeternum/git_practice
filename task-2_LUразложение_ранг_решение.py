import numpy as np

size = 3
#A = np.eye(size)
#while np.linalg.det(A) != 0:
#    A = np.random.randint(0, 20, (size,size)).astype(float)
A = np.array([[2, 1, 4], [3, 2, 1], [1, 3, 3]]).astype(float)
    
print('A:')
print(A)
print('Det(A) =', np.linalg.det(A))
print()

#Векторы перестановок
row_perms = [i for i in range(len(A))]
col_perms = [i for i in range(len(A))]

#def zero(x):
#    epsilon = 1.0e-16
#    if x < epsilon:
#        True
#    else:
#        False

#Функция замены строчек
def swap_rows(matrix, row1, row2):
    if row1 != row2:
        tmp = np.copy(matrix[row1,:])
        matrix[row1,:] = matrix[row2,:]
        matrix[row2,:] = tmp
        t = (row_perms[row1], row_perms[row2])
        row_perms[row1] = t[1]
        row_perms[row2] = t[0]
        
#Функция замены столбцов
def swap_cols(matrix, col1, col2):
    if col1 != col2:
        tmp = np.copy(matrix[:,col1])
        matrix[:,col1] = matrix[:,col2]
        matrix[:,col2] = tmp
        t = (col_perms[col1], col_perms[col2])
        col_perms[col1] = t[1]
        col_perms[col2] = t[0]
        
#Ищем максимум по всей матрице    
def find_max(matrix, index):
    maximum = matrix[index][index]
    row = index
    col = index
    for i in range(index, len(matrix)):
        for j in range(index, len(matrix)):
            if abs(matrix[i][j]) > abs(maximum):
                maximum = matrix[i][j]
                row = i
                col = j
    swap_rows(matrix, index, row)
    swap_cols(matrix, index, col)
    return matrix

def lu(matrix):
    rank = 0
    for i in range(len(matrix)-1):
        matrix = find_max(matrix, i)
        lead = matrix[i][i]
        rank = rank + 1
        if lead == 0:
            break
        for j in range(i+1, len(matrix)):
            matrix[j][i] = matrix[j][i] / lead
            coeff = matrix[j][i]
            for k in range(i+1, len(matrix)):
                matrix[j][k] = matrix[j][k] - matrix[i][k] * coeff
    L = np.zeros((len(matrix),len(matrix)))
    U = np.zeros((len(matrix),len(matrix)))
    for i in range(len(matrix)):
        for j in range(i):
            L[i][j] = matrix[i][j]
        L[i][i] = 1
        U[i][i] = matrix[i][i]
        for j in range(i+1,len(matrix)):
            U[i][j] = matrix[i][j]
    return L, U, rank

print('LU-разложение:')
L, U, rank = lu(np.copy(A))
print('Матрица L:')
print(L)
print('Матрица U:')
print(U)
print()
print('Произведение L и U:')
print(np.dot(L,U))
print()
print('Ранг =', rank)
print()

#b = np.random.randint(0, 10, (size,1)).astype(float)
b = np.array([[16], [10], [16]])
print('Вектор b:')
print(b)

#Вектор перестановок для x
col_perms_new = np.zeros(len(col_perms))
for i in range(len(col_perms)):
    col_perms_new[col_perms[i]] = i

#Решение СЛАУ
def solve_slae(L, U, b):
    global rank
    epsilon = 1.0e-10
    y = []
    Pb = np.random.randint(0, 10, (len(b),1)).astype(float)
    #Вектор b с перестановками
    for i in range(len(Pb)):
        Pb[i] = b[row_perms[i]]
    print('Вектор b с перестановками:')
    print(Pb)    
    for i in range(len(Pb)):
        sum = Pb[i]
        for j in range(i):
            sum = sum - L[i][j]*y[j]
        y.append(sum)
    if all(abs(y)<epsilon for y in y[rank:]):
        print('y =', np.array(y))
    else:
        print('Система не разрешима!')
    
    z = [0]*len(U)
    for i in reversed(range(rank)):
        sum = y[i]
        for j in range(i+1, rank):
            sum = sum - U[i][j]*z[j]
        z[i] = (sum/U[i][i])
        
    #print('z =', z)
    
    x = np.zeros(len(z))
    
    global col_perms_new
    
    for i in range(len(x)):
        x[i] = z[int(col_perms_new[i])]
    
    return x
        
x = solve_slae(L, U, b)
print('x =', x)

def check_slae(matrix, x):
    print(np.dot(matrix,x))
    
print('Проверка Ax = b:')
check_slae(A, x)

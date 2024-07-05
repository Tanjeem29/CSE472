import numpy as np

# def get_invertible_matrix_1(n):
#     mat = np.random.randint(low=-100, high=100, size=(n, n))
#     while np.linalg.matrix_rank(mat) != n:
#         mat = np.random.randint(low=-100, high=100, size=(n, n))
#     return mat

# def get_invertible_matrix_2(n):
#     mat = np.random.randint(low=-100, high=100, size=(n, n))
#     row_sums = np.sum(np.abs(mat), axis=1)
#     for i in range(n):
#         mat[i,i] = row_sums[i] - np.abs(mat[i,i]) + 1
#     return mat

def get_invertible_matrix_3(n):
    mat = np.random.randint(low=-100, high=100, size=(n, n))
    row_sums = np.sum(np.abs(mat), axis=1)
    row_sums = row_sums - np.abs(np.diag(mat)) + 1
    np.fill_diagonal(mat, row_sums)
    return mat

def reconstruct_matrix(eigenvalues, eigenvectors):
    diag = np.diag(eigenvalues)
    ans = np.matmul(np.matmul(eigenvectors, diag), np.linalg.inv(eigenvectors))
    ans2 = np.around(ans, decimals=2)
    ans3 = np.real(ans)
    return ans

def pretty_matrix(matrix):
    return np.real(np.around(matrix, decimals=2))

def test():
    n = int(input("Enter the size of the square matrix, n: "))
    m = get_invertible_matrix_3(n)
    print(f'My matrix:\n {m}\n')
    print(f'The determinant: {np.linalg.det(m)}\n')
    
    
    eigenvalues, eigenvectors = np.linalg.eig(m) 
    print(f'Eigen Values:\n {eigenvalues}\n')
    print(f'Eigen Vectors:{eigenvectors}\n')
    
    
    recon_m = reconstruct_matrix(eigenvalues, eigenvectors) 
    print(f'Recontructed matrix:\n{recon_m}\n')
    print(f'Pretty recontructed matrix:\n{pretty_matrix(recon_m)}\n')
    
    
    print(f'Is the reconstructed matrix equal to the original matrix? {np.allclose(m, recon_m)}\n')
    
    
test()




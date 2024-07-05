import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize(A, width):
    og_h, og_w = A.shape[:2]
    aspect = og_h/og_w
    new_w = width
    new_h = int(new_w*aspect)
    return cv2.resize(A, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def low_rank_approx_2d(A, k):
    U,S,V = np.linalg.svd(A)
    S = np.diag(S)
    if k > S.shape[0] or k > S.shape[1]:
        print("k is greater than the rank of the matrix")
        return
    S_k = S[:k, :k]
    U_k = U[:, :k]
    V_k = V[:k, :]
    return np.matmul(np.matmul(U_k,S_k),V_k).astype(np.uint8)

def draw(A):
    ks = [1, 5, 10, 20, 40, 50, 60, 65, 70, 80, 150, 1000]
    fig, axs = plt.subplots(3, 4, figsize=(30, 28))
    step_diff = A.shape[1]//5
    for i in range(3):
        for j in range(4):
            idx = i*4+j
            curr_img = low_rank_approx_2d(A, ks[idx])
            axs[i, j].imshow(curr_img, cmap='gray')
            axs[i, j].set_title(f'k = {ks[idx]}')
            axs[i, j].set_xticks(np.arange(0, curr_img.shape[1]+1, step=step_diff))
            axs[i, j].set_yticks(np.arange(0, curr_img.shape[0]+1, step=step_diff))
    
    plt.tight_layout()
    plt.savefig('output.png')
    plt.show()
    
  
def test(filename):
    img = cv2.imread(filename)
    img_gray_2d=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img_gray_2d = resize(img_gray_2d, 1000)
    draw(resized_img_gray_2d)
    print('Lowest k is around 65\n')

test('image.jpg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import imageio
import os


def load_data(input_file_name = None):
    if input_file_name is None:
        input_file_name = input("Enter the name of the input file: ")
    input_file = './Data/' + input_file_name
    # Get each point as lines
    with open(input_file, "r") as f:
        data = f.read().splitlines()

    
    # Split at ',' to get list of each dimension
    data = [i.split(",") for i in data]

    # i = each point
    # j = each dimension
    # Convert each dimension to float
    data = [[float(j) for j in i] for i in data]
    data = np.array(data)
    print(data.shape)
    return data

def plot_2d(data_2D):
    plt.figure()
    plt.scatter(data_2D[:,0], data_2D[:,1])
    plt.show()
    
def save_2d(data_2D, filename):
    plt.figure()
    plt.scatter(data_2D[:,0], data_2D[:,1])
    plt.title(f'PCA for {filename}')
    plt.savefig(filename + '_pca.png')
    plt.close()

def pca_generate(data, k, greaterthan2=True, filename=None):
    # columns are features, rows are datapoints
    if greaterthan2 and data.shape[1] == 2:

        if filename is not None:
            base_filename = filename.split('.')[0]
            results_dir = f'./Results/{base_filename}'
            os.makedirs(results_dir, exist_ok=True)
            filename = f'{results_dir}/{base_filename}'
            save_2d(data, filename)
        return data
    row_means = data.mean(axis=0)

    data = data - row_means

    U,S,VT = np.linalg.svd(data, full_matrices=False)
    V = VT.T
    pca = np.matmul(data, V[:,:k])

    if filename is not None:
        base_filename = filename.split('.')[0]
        results_dir = f'./Results/{base_filename}/{k}'
        os.makedirs(results_dir, exist_ok=True)
        filename = f'{results_dir}/{base_filename}'
        save_2d(pca, filename)
    return pca
    

class gmm:
    # ith gaussian compoonent, jth datapoint
    def __init__(self, data, k, filename=None):
        self.file_name = filename
        self.data = data
        self.k = k
        self.total_data = data.shape[0]
        self.num_features = data.shape[1]
        
             
        self.w = np.ones(k) / k
        self.mu = self.data[np.random.choice(self.total_data, k, replace=False)]
        self.sig = np.array([np.eye(self.num_features) for i in range(k)])
        
        self.clusters = None
        
    def E_step(self):
        self.p = np.zeros((self.k, self.total_data))
        for k in range(self.k):
            
            pdf = multivariate_normal(self.mu[k], self.sig[k], allow_singular=True)
            self.p[k] = (self.w[k] * pdf.pdf(self.data))
        
        self.p = self.p / (np.sum(self.p, axis=0)+ 1e-10)
    
    
    # p -> clusters x datapoints
    # mu -> clusters x features
    # data = datapoints x features
    # ith cluster, jth datapoint
    
    def M_step(self):
        self.n = np.sum(self.p, axis=1)    #1 == collapse colmn
        self.w = self.n / self.total_data
        for i in range(self.k):
            self.mu[i] = (self.p[i] @ self.data) / (self.n[i] + 1e-10)
            diff = self.data - self.mu[i]  
            weighted_diff = self.p[i].reshape(-1,1) * diff  
            self.sig[i] = weighted_diff.T @ diff / (self.n[i] + 1e-10)
        
    def train(self, max_iter=5000, tol=1e-1, trial=0, mode = 3):
        gif_img_files = []
        prev_ll = -np.inf
        for i in range(max_iter):
            self.E_step()
            self.M_step()
            ll = self.log_likelihood()
            img_name = f'{self.file_name}_{self.k}_{trial}_{i}_.png'
            self.plot_clusters(img_name, mode=mode)
            if mode == 1 or mode == 2:
                gif_img_files.append(img_name)
            if np.abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        if mode == 1 or mode == 2:
            imageio.mimsave(f'{self.file_name}_{self.k}_{trial}.gif', [imageio.imread(f) for f in gif_img_files], fps=2)
        for img_file in gif_img_files:
            os.remove(img_file)
        return ll
        
    def log_likelihood(self):
        ans = 0
        for k in range(self.k):
            pdf = multivariate_normal(self.mu[k], self.sig[k], allow_singular=True)
            ans += self.w[k] * pdf.pdf(self.data) + 10e-10 
        return np.sum(np.log(ans) + 1e-10)
    
    
    def clusterize(self):
        self.clusters = np.argmax(self.p, axis=0).flatten()
        return self.clusters
    
    def draw_ellipse(self, cluster_idx, num_ellipses=4):
        mean = self.mu[cluster_idx]
        cov = self.sig[cluster_idx]

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        for n in range(1, num_ellipses + 1):

            scaled_eigenvalues = eigenvalues * (2 * n)
            width, height = 2 * np.sqrt(scaled_eigenvalues)

            # Draw the ellipse
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='black', fc='None', lw=2)
            plt.gca().add_patch(ellipse)
    
    def plot_clusters(self, filename=None, mode = 0): #0-> show, 1->save, 2->both, 3-> no plot
        self.clusterize()
        plt.figure(figsize=(8, 6))
        for i in range(self.k):
            cluster_data = self.data[self.clusters == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
            self.draw_ellipse(i)
        
        plt.title("GMM Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        if filename is not None and (mode == 1 or mode == 2):
            plt.savefig(filename)
        if mode == 0 or mode == 2:
            plt.show()
        else :
            plt.close()
    
    def describe_gmm(self):
        print("Number of clusters: ", self.k)
        print("Means: ", self.mu)
        print("Log Likelihood: ", self.log_likelihood())

        

def trials_same_k(data, k, trials=10, filename=None, mode=1):
    base_filename = filename.split('.')[0]
    results_dir = f'./Results/{base_filename}/{k}'
    os.makedirs(results_dir, exist_ok=True)
    filename = f'{results_dir}/{base_filename}'
    ll = -np.inf
    for i in range(trials):
        print("*********** Trial = ", i)
        mygmm = gmm(data, k, filename=filename)
        mygmm.train(trial=i, mode=mode)
        gif_name = f'{results_dir}/{base_filename}_{k}_{i}.gif'
        
        if mygmm.log_likelihood() > ll:
            ll = mygmm.log_likelihood()
            best_gmm = mygmm
            if mode == 1 or mode == 2:
                if os.path.exists(f'{results_dir}/{base_filename}_{k}_best.gif'):
                    os.remove(f'{results_dir}/{base_filename}_{k}_best.gif')
                os.rename(gif_name, f'{results_dir}/{base_filename}_{k}_best.gif')
        else:
            os.remove(gif_name)
    print("*********** Best ll = ", ll)
    return best_gmm


def trials_diff_k(data, max_k, trials=10, mode = 0, filename=None):
    ll = -np.inf
    best_ll_per_k = []
    k_values = [i for i in range(3,max_k+1)]
    for i in k_values:
        print("*********** k = ", i)
        k_best_gmm = trials_same_k(data, i, trials, mode = mode, filename=filename)
        base_filename = filename.split('.')[0]
        plot_name = f'./Results/{base_filename}/{i}/{base_filename}_{i}_best.png'
        k_best_gmm.plot_clusters(plot_name, mode=1)
        if k_best_gmm.log_likelihood() > ll:
            ll = k_best_gmm.log_likelihood()
            best_gmm = k_best_gmm
        best_ll_per_k.append(ll)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, best_ll_per_k, marker='o')
    plt.title(f'Best Log-Likelihood for Each k - Dataset: {filename}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    if filename is not None:
        base_filename = filename.split('.')[0]
        plt.savefig(f'./Results/{base_filename}/best_ll_per_k.png')
    plt.show()
    
    return best_gmm



def run():
    filename = input("Enter the name of the input file: ")
        
    print('Data Loading')
    temp = load_data(filename)
    print('Data Loaded')
    
    print('PCA Started')
    pca = pca_generate(temp, 2, filename=filename)
    print('PCA Done')
    
    trials = int(input("Enter the number of trials per cluster: "))
    
    print('Single or Ranged k?')
    flag = int(input("Enter 1 for single k, 2 for range of k: "))
    if flag == 1:
        k = int(input("Enter k: "))
        mode = int(input("Enter 0 for plot, 1 for save, 2 for both: "))
        print('Single k Started')
        bestgmm = trials_same_k(pca, k, trials, filename, mode=mode)
        print('Single k Done')
    elif flag == 2:
        max_k = int(input("Enter max k: "))
        mode = int(input("Enter 0 for plot, 1 for save, 2 for both: "))
        print('Ranged k Started')
        bestgmm = trials_diff_k(pca, max_k, trials, filename=filename, mode=mode)
        print('Ranged k Done')
    else:
        print('Invalid Input')
        return
    
run()


# 6D_data_points.txt
# 10
# 1
# 5
# 1
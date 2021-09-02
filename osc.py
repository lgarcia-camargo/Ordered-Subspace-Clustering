
import numpy as np
from numpy import linalg as lin



class osc():

    def __init__(self, max_iters, lambda_1, lambda_2, mu, diagconstraint):
        self.max_iterations = max_iters
        self.lambda_1       = lambda_1
        self.lambda_2       = lambda_2
        self.mu             = mu
        self.diagconstraint = diagconstraint


    def solve_l2(self, w, lam):

        #solving: arg min_{x} 1/2 || x - b ||_2^2 + lambda/2 || x ||_2^2

        return np.divide(w, lam+1)


    def solve_l1(self, b, lam):

        #solving: arg min_{x} || x - b ||_2^2 + lambda || x ||_1

        x = np.abs(b)-lam
        x[x<0] = 0

        return np.multiply(np.sign(b), x)


    def solve_l1l2(self, w, lam):
        m, n = w.shape

        E = w

        for i in range(n):
            norm_col = lin.norm(w[:,i])

            if norm_col > lam:
                E[:,i] = (norm_col-lam) * w[:,i] / norm_col

            else:
                E[:,i] = np.zeros(m)

        
        return E

    def norm_l1l2(self, x):
        L = 0
        for i in range(x.shape[1]):
            L += lin.norm(x[:,i])
        
        return L

    def normalize(self, X):
        X = X.astype('float64')
        for i in range(X.shape[1]):
            X[:, i] = X[:,i]/(np.matmul(X[:,i].transpose(), X[:,i]))**.5
        return X

    def osc_exact(self, X):
        
        X = self.normalize(X)
        mu = self.mu

        func_values    = np.zeros((self.max_iterations))
        xm, xn = X.shape

        #creating coefficient matrix
        Z = np.zeros((xn,xn))
        Z_prev = Z

        #creating gaussian noise matrix
        E = np.zeros((xm, xn))
        E_prev = E

        #creating penalty matrix for sequential nature of data
        R = np.triu(np.ones((xn, xn-1)),1) - np.triu(np.ones((xn, xn-1))) + np.triu(np.ones((xn, xn-1)),-1) - np.triu(np.ones((xn, xn-1)))


        J = np.zeros((xn, xn-1))
        J_prev = J

        Y_1 = np.zeros((xm,xn))
        Y_2 = np.zeros((xn,xn-1))

        mu_max =  10
        gamma_0 = 1.1


        normfX = lin.norm(X,'fro')
        rho = (lin.norm(X)**2) * 1.1


        tol_1 = 1*10**-2
        tol_2 = 1*10**-4


        
        for k in range(self.max_iterations):
             
            #update Z
            partial = mu * (np.matmul(X.transpose(),(np.matmul(X,Z_prev)- (X-E_prev - 1/mu * Y_1)))  + np.matmul((np.matmul(Z_prev, R) - (J_prev + 1/mu * Y_2)), R.transpose()))

            V = Z_prev  -  1/rho* partial


            Z = self.solve_l1(V, self.lambda_1/rho)


            #set diagonal to 0
            if self.diagconstraint:
                Z = np.fill_diagonal(Z, 0)

            #update E
            V = np.matmul(-X,Z_prev) + X - 1/mu *Y_1
            E = self.solve_l2(V, 1/mu)

            #update J
            J= self.solve_l1l2(np.matmul(Z_prev,R) - 1/mu * Y_2, self.lambda_2/mu)
            

            Y_1 += mu * (np.matmul(X, Z) - X + E)
            Y_2 += mu * (J - np.matmul(Z, R))

            #update mu
            if mu * np.sqrt(rho) * (max([lin.norm(Z - Z_prev,'fro') / normfX, lin.norm(E - E_prev, 'fro') / normfX, lin.norm(J-J_prev,'fro') / normfX])) < tol_2:
                gamma = gamma_0
            
            else:
                gamma = 1



            mu = min(mu_max, gamma*mu)

            func_values[k] = .5 * lin.norm(E, 'fro')**2 + self.lambda_1*lin.norm(Z, 1) + self.lambda_2 * self.norm_l1l2(np.matmul(Z,R))
            

            if lin.norm(np.matmul(X,Z)-X +E, 'fro')/normfX < tol_1 and lin.norm(J- np.matmul(Z,R), 'fro')/normfX < tol_1 and (mu * rho**.5 * max([lin.norm(Z-Z_prev,'fro'), lin.norm(E-E_prev, 'fro'), lin.norm(J-J_prev, 'fro')])/normfX < tol_2):
                break


            Z_prev = Z
            E_prev = E
            J_prev = J

        return Z





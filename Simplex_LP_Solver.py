import numpy as np # type: ignore

def initialize_matrices(A, b, c, num_vars, num_eqns):
    x = np.ones(num_eqns)
    y = np.ones(num_vars)
    s = np.ones(num_eqns)

    temp_matrix = np.eye(num_vars)  
    return x, y, s

def compute_diagonal_matrices(x, s):
 
    return np.diag(x), np.diag(s)

def construct_augmented_system_matrix(A, num_vars, num_eqns, S, X):

    zero_matrix = np.zeros((num_vars, num_vars))  
    zero_nv_ne = np.zeros((num_vars, num_eqns))
    zero_ne_ne = np.zeros((num_eqns, num_eqns))
    identity_ne = np.eye(num_eqns)

  
    upper_block = np.hstack([A, zero_matrix, zero_nv_ne])
    middle_block = np.hstack([zero_ne_ne, A.T, identity_ne])
    lower_block = np.hstack([S, zero_nv_ne.T, X])

   
    return np.vstack([upper_block, middle_block, lower_block])

def construct_rhs_vector(b, A, x, c, y, s, mu, num_eqns):
    e = np.ones(num_eqns)
    sigma = 0.2  
    X, S = compute_diagonal_matrices(x, s)
    XS_product = np.dot(X, S.dot(e))

    
    r1 = b - np.dot(A, x)
    r2 = c - np.dot(A.T, y) - s
    r3 = sigma * mu * e - XS_product

    return np.concatenate([r1, r2, r3])

def find_step_size(x, s, delta_x, delta_s):
   
    alpha_x = min([x[i] / -delta_x[i] for i in range(len(x)) if delta_x[i] < 0], default=np.inf)
    alpha_s = min([s[i] / -delta_s[i] for i in range(len(s)) if delta_s[i] < 0], default=np.inf)

    
    eta = 0.999
    return eta * min(alpha_x, alpha_s)

def check_convergence(x, s, mu, epsilon):

    primal_dual_gap = np.dot(x, s)
    is_feasible = np.all(x > 0) and np.all(s > 0)
    is_converged = primal_dual_gap < epsilon and is_feasible
    return is_converged

def interior_point(A, b, c, num_vars, num_eqns,epsilon):
    x, y, s = initialize_matrices(A, b, c, num_vars, num_eqns)
    mu = np.dot(x, s) / num_eqns

    while not check_convergence(x, s, mu, epsilon):
        X, S = compute_diagonal_matrices(x, s)
        A_dash = construct_augmented_system_matrix(A, num_vars, num_eqns, S, X)
        b_dash = construct_rhs_vector(b, A, x, c, y, s, mu, num_eqns)
        delta = np.linalg.solve(A_dash, b_dash)
        
        delta_x, delta_y, delta_s = delta[:num_eqns], delta[num_eqns:num_vars+num_eqns], delta[num_vars+num_eqns:]
        
        alpha = find_step_size(x, s, delta_x, delta_s)
       
        x += alpha * delta_x
        y += alpha * delta_y
        s += alpha * delta_s
        mu = np.dot(x, s) / num_eqns

    print("Optimal x:", x)
    print("Optimal value:", np.dot(c.T, x))
    return x

if __name__ == '__main__':
    #testcasesse

    	# A = np.array([[1,1,1,0],[3,2,0,1]])
	# b = np.array([5,12])
	# # b = b.reshape(b.shape[0],1)
	# c = np.array([-6,-5,0,0])
	# x = np.zeros(A.shape[1])


    c = np.array([2, 3, 1, 4, 5, 2, 1, 3, 4, 2])
    A = np.array([
    [1, 0, 1, 2, 0, 1, 0, 3, 2, 1],
    [0, 2, 1, 1, 1, 0, 2, 0, 1, 2],
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
])
    b = np.array([5, 6, 4])

#     c = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# A= np.array([
#     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# 	])
# b = np.array([3, 4])

 
#     c = np.array([3, 2, 1, 4, 5, 2, 1, 3, 4, 2])
# A = np.array([
#     [1, 2, 1, 0, 1, 0, 1, 2, 1, 1],
#     [0, 1, 2, 1, 1, 0, 2, 0, 1, 2],
#     [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [2, 0, 1, 1, 2, 1, 1, 0, 1, 2]
# ])
# b = np.array([8, 6, 7, 10])


#     c = np.array([2, 3, 1, 4, 5, 2, 1, 3, 4, 2])
# A = np.array([
#     [1, 0, 1, 2, 0, 1, 0, 3, 2, 1],
#     [0, 2, 1, 1, 1, 0, 2, 0, 1, 2],
#     [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [2, 1, 0, 1, 1, 0, 1, 1, 0, 1]
# ])
# b = np.array([5, 6, 4, 7])



#     c = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# A = np.array([
#     [1, 2, 1, 0, 1, 0, 1, 2, 1, 1],
#     [0, 1, 2, 1, 1, 0, 2, 0, 1, 2],
#     [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     [2, 0, 1, 1, 2, 1, 1, 0, 1, 2],
#     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
# ])
# b = np.array([8, 6, 7, 10, 4])
    # A = np.array([[10, 5, 1, 0, 0, 0],
    #               [2, 3, 0, 1, 0, 0],
    #               [1, 0, 0, 0, 1, 0],
    #               [0, 1, 0, 0, 0, 1]])
    # c = np.array([-1000, -1200, 0, 0, 0, 0])
    # b = np.array([200, 60, 34, 14])
    epsilon = 1e-10
interior_point(A, b, c, A.shape[0], A.shape[1],epsilon)

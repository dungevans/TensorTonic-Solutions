import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:    
    y_train = np.asarray(y_train)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    N_total = y_train.shape[0]  
    
    classes = np.sort(np.unique(y_train))
    
    thetas = {}
    priors = {}
    
    for c in classes:
        X_c_samples = X_train[y_train == c]
        N_c = X_c_samples.shape[0]
        
        priors[c] = np.log(N_c / N_total)
        
        N_jc = np.sum(X_c_samples, axis=0)
        
        thetas[c] = (N_jc + 1) / (N_c + 2)
            
    Y_predict = []
    for x in X_test:
        sample_scores = []
        
        for c in classes:
            theta_c = thetas[c]
            point_c = priors[c]
            
            p_c = point_c + np.sum(x * np.log(theta_c)) + np.sum((1 - x) * np.log(1 - theta_c))
            sample_scores.append(round(p_c, 4))
            
        Y_predict.append(sample_scores)    
        
    return np.asarray(Y_predict)
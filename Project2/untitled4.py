# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:08:35 2019

@author: Michael
"""

    best_unit = np.argmin()
    
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, item), #M features to n_hidden_units
            torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.Linear(item, 1), # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
            )    
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_ANN,
                                                       y=y_train_ANN,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test_ANN)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test_ANN.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 

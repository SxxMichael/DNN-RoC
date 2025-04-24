import Sim
import tensorflow as tf
import numpy as np

def main(n, Epoch, L, P, type, distribution, sim_loop = 500, lr = 0.001, sigma = 1):
    cost_Arr = []
    actValue_Arr = []
    sVal_Arr = []

    for i in range(sim_loop):
        X, Y, tru_F = Sim.test_data(10, n, sigma, type, distribution)
        model = tf.keras.Sequential()

        for j in range(L):
            model.add(tf.keras.layers.Dense(P, activation="relu"))

        model.add(tf.keras.layers.Dense(1))
        opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.fit(X.T, Y.T, epochs=Epoch, batch_size=n, verbose = 0)

        val_pred = model.predict(X.T, verbose = 0)
        cost = (1 / n) * np.dot((Y - val_pred.T), (Y - val_pred.T).T)
        cost_Arr.append(cost)

        act_val = (1 / n) * np.dot((tru_F - val_pred.T), (tru_F - val_pred.T).T)
        actValue_Arr.append(act_val)

        sample_Var = np.var(Y)
        sVal_Arr.append(sample_Var)

    cost_Arr = np.reshape(cost_Arr, (1, sim_loop)) 
    actValue_Arr = np.reshape(actValue_Arr, (1, sim_loop))
    sVal_Arr = np.reshape(sVal_Arr, (1, sim_loop))

    f_Arr = np.vstack((cost_Arr, actValue_Arr, sVal_Arr))
    return f_Arr
        

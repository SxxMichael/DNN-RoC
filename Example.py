from DNN_RoC import *

res = main(n=3000, Epoch=2000, L=2, P=14, type="f3",distribution="unif", sim_loop=500, lr=0.01)

for row in res.T:
    print(' '.join(map(str, row)))

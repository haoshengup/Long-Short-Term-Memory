import lstm
net = lstm.NetWork([2, 100, 1])
net.train(0.1, 10000)
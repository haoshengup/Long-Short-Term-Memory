import numpy as np

class NetWork(object):
    def __init__(self, sizes):
        self.sizes = sizes

        # initialize input gate
        self.w_xi = np.random.randn(self.sizes[1], self.sizes[0])/np.sqrt(self.sizes[0])
        self.w_hi = np.random.randn(self.sizes[1], self.sizes[1])/np.sqrt(self.sizes[1])
        self.w_ci = np.random.randn(self.sizes[1], self.sizes[1])/np.sqrt(self.sizes[1])

        # initialize derivative of input gate
        self.w_xi_derivative = np.zeros_like(self.w_xi)
        self.w_hi_derivative = np.zeros_like(self.w_hi)
        self.w_ci_derivative = np.zeros_like(self.w_ci)

        # initialize forget gate
        self.w_xf = np.random.randn(self.sizes[1], self.sizes[0])/np.sqrt(self.sizes[0])
        self.w_hf = np.random.randn(self.sizes[1], self.sizes[1]) / np.sqrt(self.sizes[1])
        self.w_cf =  np.random.randn(self.sizes[1], self.sizes[1]) / np.sqrt(self.sizes[1])

        # initialize derivative of forget gate
        self.w_xf_derivative = np.zeros_like(self.w_xf)
        self.w_hf_derivative = np.zeros_like(self.w_hf)
        self.w_cf_derivative = np.zeros_like(self.w_cf)

        # intialize cell state
        self.w_xc = np.random.randn(self.sizes[1], self.sizes[0])/np.sqrt(self.sizes[0])
        self.w_hc = np.random.randn(self.sizes[1], self.sizes[1]) / np.sqrt(self.sizes[1])

        # initialize derivative of cell state
        self.w_xc_derivative = np.zeros_like(self.w_xc)
        self.w_hc_derivative = np.zeros_like(self.w_hc)

        # initialize output gate
        self.w_xo = np.random.randn(self.sizes[1], self.sizes[0])/np.sqrt(self.sizes[0])
        self.w_ho = np.random.randn(self.sizes[1], self.sizes[1]) / np.sqrt(self.sizes[1])
        self.w_co = np.random.randn(self.sizes[1], self.sizes[1]) / np.sqrt(self.sizes[1])

        # initialize derivative of output gate
        self.w_xo_derivative = np.zeros_like(self.w_xo)
        self.w_ho_derivative = np.zeros_like(self.w_ho)
        self.w_co_derivative = np.zeros_like(self.w_co)

        # initialize out weights and derivative
        self.w = np.random.randn(self.sizes[2], self.sizes[1]) / np.sqrt(self.sizes[1])
        self.w_derivative = np.zeros_like(self.w)

        # a list to preserve binary data
        binary_dim = 8
        self.largest_num = 2 ** binary_dim
        binary = np.unpackbits(np.array(list(range(self.largest_num)),dtype=np.uint8)[:,np.newaxis], axis=1)
        self.int2binary = [binary[i,:] for i in range(self.largest_num)]

    def forward(self, x):
        step = x.shape[1]
        i_gate = np.zeros(shape=(step, self.sizes[1], 1))
        f_gate = np.zeros(shape=(step + 1, self.sizes[1], 1))
        net_ct = np.zeros(shape=(step, self.sizes[1], 1))
        c_state = np.zeros(shape=(step + 1, self.sizes[1], 1))
        o_gate = np.zeros(shape=(step, self.sizes[1], 1))
        h_state = np.zeros(shape=(step + 1, self.sizes[1], 1))
        layer_out = np.zeros(shape=(step, 1))

        for position in range(step):
            i_gate[position] = self.sigmoid(np.dot(self.w_xi, x[:,position][:,np.newaxis]) + np.dot(self.w_hi, h_state[position]) \
                                + np.dot(self.w_ci, c_state[position]))
            f_gate[position] = self.sigmoid(np.dot(self.w_xf, x[:,position][:,np.newaxis]) + np.dot(self.w_hf, h_state[position]) \
                                + np.dot(self.w_cf, c_state[position]))
            net_ct[position] = np.dot(self.w_xc, x[:,position][:,np.newaxis]) + np.dot(self.w_hc, h_state[position])
            c_state[position + 1] = f_gate[position] * c_state[position] + i_gate[position] * self.tanh(net_ct[position])
            o_gate[position] = self.sigmoid(np.dot(self.w_xo, x[:,position][:,np.newaxis]) + np.dot(self.w_ho, h_state[position]) \
                                + np.dot(self.w_co, c_state[position + 1]))
            h_state[position + 1] = o_gate[position] * self.tanh(c_state[position + 1])
            layer_out[position] = self.sigmoid(np.dot(self.w, h_state[position + 1]))

        return i_gate, f_gate, c_state, o_gate, h_state, net_ct, layer_out

    def backward(self, x, out, target):
        step = len(target)
        # calculate error of output
        delta_layer_out = [self.cost_derivative(out[-1][i], target[i]) * self.sigmoid_derivative(out[-1][i]) for i in range(step)]

        # calculate error of states
        delta_i_gate = np.zeros(shape=(step + 1, self.sizes[1], 1))
        delta_f_gate = np.zeros(shape=(step + 1, self.sizes[1], 1))
        delta_c_state = np.zeros(shape=(step + 1, self.sizes[1], 1))
        delta_o_gate = np.zeros(shape=(step, self.sizes[1], 1))
        delta_h_state = np.zeros(shape=(step + 1, self.sizes[1], 1))

        for i in reversed(list(range(step))):
            delta_h_state[i] = np.dot(self.w.T, delta_layer_out[i])[:,np.newaxis] + np.dot(self.w_hc.T, delta_h_state[i + 1])                                
            delta_o_gate[i] = delta_h_state[i] * self.sigmoid_derivative(out[3][i]) * self.tanh(out[2][i + 1])
            delta_c_state[i] = delta_h_state[i] * self.tanh_derivative(self.tanh(out[2][i + 1])) * out[3][i] + out[1][i + 1] * delta_c_state[i + 1] \
                                + np.dot(self.w_ci.T, delta_i_gate[i + 1]) + np.dot(self.w_cf.T, delta_f_gate[i + 1]) \
                                + np.dot(self.w_co.T, delta_o_gate[i])
            delta_f_gate[i] = self.sigmoid_derivative(out[1][i]) * delta_c_state[i] * out[2][i]
            delta_i_gate[i] = self.sigmoid_derivative(out[0][i]) * delta_c_state[i] * self.tanh(out[5][i])

        # calculate derivative of output gate
        self.w_xo_derivative = sum([np.dot(delta_o_gate[i], x[:,i][:,np.newaxis].T) for i in range(step)])
        self.w_ho_derivative = sum([np.dot(delta_o_gate[i], out[4][i].T) for i in range(step)])
        self.w_co_derivative = sum([np.dot(delta_o_gate[i], out[2][i + 1].T) for i in range(step)])

        # calculate derivative of cell state
        self.w_xc_derivative = sum([np.dot(delta_c_state[i], x[:, i][:, np.newaxis].T) * out[0][i] * self.tanh_derivative(self.tanh(out[5][i])) for i in range(step)])
        self.w_hc_derivative = sum([np.dot(delta_c_state[i], out[4][i].T) * out[0][i] * self.tanh_derivative(self.tanh(out[5][i])) for i in range(step)])

        # calculate derivative of forget gate
        self.w_xf_derivative = sum([np.dot(delta_f_gate[i], x[:, i][:, np.newaxis].T) for i in range(step)])
        self.w_hf_derivative = sum([np.dot(delta_f_gate[i], out[4][i].T) for i in range(step)])
        self.w_cf_derivative = sum([np.dot(delta_f_gate[i], out[2][i].T) for i in range(step)])

        # calculate derivative of input gate
        self.w_xi_derivative = sum([np.dot(delta_i_gate[i], x[:, i][:, np.newaxis].T) for i in range(step)])
        self.w_hi_derivative = sum([np.dot(delta_i_gate[i], out[4][i].T) for i in range(step)])
        self.w_ci_derivative = sum([np.dot(delta_i_gate[i], out[2][i].T) for i in range(step)])

        # calculate derivative of out weights
        self.w_derivative = sum([np.dot(delta_layer_out[i], out[4][i + 1].T) for i in range(step)])

    def train(self, beta, epoches):
        for j in range(epoches):
            a_int = np.random.randint(self.largest_num/2)
            a = self.int2binary[a_int]

            b_int = np.random.randint(self.largest_num/2)
            b = self.int2binary[b_int]

            c_int = a_int + b_int
            c = self.int2binary[c_int]

            d_int = 0

            x = np.zeros(shape=(2, len(a)))
            x[0] = a[::-1]
            x[1] = b[::-1]

            out = self.forward(x)
            self.backward(x, out, c[::-1])

            self.w_xi -= beta * self.w_xi_derivative
            self.w_hi -= beta * self.w_hi_derivative
            self.w_ci -= beta * self.w_ci_derivative

            self.w_xf -= beta * self.w_xf_derivative
            self.w_hf -= beta * self.w_hf_derivative
            self.w_cf -= beta * self.w_cf_derivative

            self.w_xc -= beta * self.w_xc_derivative
            self.w_hc -= beta * self.w_hc_derivative

            self.w_xo -= beta * self.w_xo_derivative
            self.w_ho -= beta * self.w_ho_derivative
            self.w_co -= beta * self.w_co_derivative

            self.w -= beta * self.w_derivative

            if j % 200 == 0:
                beta *= 0.9
            d = np.round(out[-1]).reshape(8).tolist()
            for index, num in enumerate(d):
                d_int += num * 2**index
			d_int = int(d_int)
			if j%500 == 0:
                print('Epoch ' + str(j) + ':   ' + str(a_int) + " + " + str(b_int) + " = " + str(d_int) + '\n')
         
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanh_derivative(self, out):
        return 1 - out ** 2

    def cost_derivative(self, out, target):
        return (out - target)

    def cost(self, out, target):
        return  sum(0.5 * sum((out - target) ** 2))
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/RTL')
if args.mode == 'server':
    sys.path.append('/home/liyang/codes/RTS')

from tlbo.model.basics.gp_reg import GPR

if __name__ == "__main__":
    x = np.random.rand(10)*10
    y = np.sin(x)
    gpr = GPR()
    X = x.reshape(-1, 1)
    gpr.train(X, y)
    res = gpr.predict(X)
    print(res)
    print(y)
    xs = np.linspace(0, 10, 500)
    ys = gpr.predict(xs.reshape((-1, 1)))
    res = ys
    plt.plot(xs, res[:, 0])
    plt.fill_between(xs, res[:, 0] - res[:, 1], res[:, 0] + res[:, 1], color='blue', alpha='0.5')
    plt.show()

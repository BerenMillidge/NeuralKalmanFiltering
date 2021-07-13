
# experiments with different learning rates and num steps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import scipy.linalg as LA
import brewer2mpl
import seaborn as sns

bmap = brewer2mpl.get_map("Set2", 'qualitative',7)
colors = bmap.mpl_colors

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [5, 5],
   'font.family': 'sans-serif'
   }
mp.rcParams.update(params)
def u_fun(t):
  return np.exp(-0.01 * t)

def run_NKF(num_iters=5, lr = 0.05):
  dt = 0.001
  A = np.array([[1, dt, 0.5 * dt**2],
      [0,1,dt],
      [0,0,1]])
  x = np.array([0,0,0])
  #C = np.identity(3)
  C = np.random.normal(0,1,[3,3])
  print(C.shape)
  print(np.dot(A,x))
  us = []
  xs = []
  ys = []
  B = np.array([0,0,1])
  xhat = np.array([0.0,0.0,0.0])
  phat = np.identity(3)
  Q = np.identity(3)
  R = np.identity(3)
  xhats = []
  phats = []
  xhat_grad = xhat
  phat_grad = np.copy(phat)
  xhat_grads = []
  xhat_grad2 = np.copy(xhat_grad)
  xhat_grads2 = []
  for i in range(1000):
    u = u_fun(i)
    x = np.dot(A,x) + np.dot(B,u) #+ np.random.normal(0,0.1,3)
    y = np.dot(C, x) + np.random.normal(0,1,3)
    xhat_proj = np.dot(A,xhat) + np.dot(B,u)
    phat_proj = np.dot(A, np.dot(phat, A.T)) + Q
    K = np.dot(np.dot(phat_proj, C.T), LA.inv(np.dot(np.dot(C,phat_proj),C.T) + R))
    xhat = xhat_proj + np.dot(K, y - np.dot(C,xhat_proj))
    phat = phat_proj - np.dot(K, np.dot(C, phat_proj))
    for i in range(num_iters):
      ex = xhat_grad - (np.dot(A, xhat_grad) + np.dot(B,u))
      ey = y - np.dot(C, xhat_grad)
      dldmu = np.dot(phat_grad, ex) - np.dot(C.T, np.dot(R, ey))
      xhat_grad -= (lr * dldmu)

    xs.append(np.copy(x))
    ys.append(y)
    xhats.append(np.copy(xhat))
    xhat_grads.append(np.copy(xhat_grad))
  xs = np.array(xs)
  xhats = np.array(xhats)
  xhat_grads = np.array(xhat_grads)
  return xs, xhats, xhat_grads

def plot_graph2(xs, xhats, xhat_grads,xhat_grads2, title, num_steps):
  plt.plot(xs,label="True Value",color=colors[4])
  plt.plot(xhats, label="Kalman Filter",color=colors[2])
  plt.plot(xhat_grads, label= str(num_steps[0]) +" Gradient Steps", linestyle='-.',color=colors[1])
  plt.plot(xhat_grads2,label=str(num_steps[1]) +" Gradient Steps", linestyle='-.', color=colors[5])
  plt.title(title)
  leg = plt.legend()
  f = leg.get_frame()
  f.set_edgecolor('1')
  f.set_facecolor('0.96')
  plt.xticks(np.arange(0,101,10))
  plt.xlabel('Timestep',fontsize=10)
  plt.ylabel("Predicted Value",fontsize=10)
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.savefig(title + "_NKF_zoomed.eps", format="eps")
  plt.show()

def plot_lr_comparison(xs, xhats, xhat_grad_list, title,learning_rates,idx):
  plt.plot(xs,label="True Value",color=colors[4])
  plt.plot(xhats, label="Kalman Filter",color=colors[2])
  for i, xhat_grads in enumerate(xhat_grad_list):
    plt.plot(xhat_grad_list[i][599:699, idx], label= str(learning_rates[i]) + " Learning Rate", linestyle='-.')
  plt.title(title)
  leg = plt.legend()
  f = leg.get_frame()
  f.set_edgecolor('1')
  f.set_facecolor('0.96')
  plt.xticks(np.arange(0,101,10))
  plt.xlabel('Timestep',fontsize=10)
  plt.ylabel("Predicted Value",fontsize=10)
  sns.despine(left=False,top=True, right=True, bottom=False)
  plt.savefig(title.replace(" ", "_") + "_NKF_zoomed.eps", format="eps")
  plt.show()

def lr_comparison(learning_rates, num_iters):
  xhat_grad_list = []
  for i in range(len(learning_rates)):
    xs, xhats, xhat_grads = run_NKF(num_iters=num_iters, lr=learning_rates[i])
    xhat_grad_list.append(xhat_grads)
  plot_lr_comparison(xs[599:699,0], xhats[599:699,0],xhat_grad_list, "Position Learning Rate Comparison", learning_rates, 0)
  plot_lr_comparison(xs[599:699,1], xhats[599:699,1],xhat_grad_list, "Velocity Learning Rate Comparison", learning_rates, 1)
  plot_lr_comparison(xs[599:699,2], xhats[599:699,2],xhat_grad_list, "Acceleration Learning Rate Comparison", learning_rates, 2)


if __name__ == '__main__':
    xs, xhats, xhat_grads = run_NKF(num_iters = 5, lr=0.07)
    xs, xhats, xhat_grads2 = run_NKF(num_iters=1, lr=0.25)

    plot_graph2(xs[599:699,0], xhats[599:699,0], xhat_grads[599:699,0],xhat_grads2[599:699,0], "Estimated Position",num_steps=[5,1])
    plot_graph2(xs[599:699,1], xhats[599:699,1], xhat_grads[599:699,1],xhat_grads2[599:699,1], "Estimated Velocity", num_steps = [5,1])
    plot_graph2(xs[599:699,2], xhats[599:699,2], xhat_grads[599:699,2],xhat_grads2[599:699,2], "Estimated Acceleration",num_steps = [5,1])

    
    learning_rates = [0.01,0.05,0.07,0.2]
    lr_comparison(learning_rates, num_iters = 3)

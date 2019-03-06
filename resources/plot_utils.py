import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() - 0.25, x.max() + 0.25
    y_min, y_max = y.min() - 0.25, y.max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, model, xx, yy, prob=False, **params):
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(xy)[:,1] if prob else model.predict(xy)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)
    return ax

def plot_decision_boundary(X, y, model, prob=False, cmap='rainbow', alpha=0.6, levels=None):
    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy, prob=prob, cmap=cmap, alpha=alpha, levels=levels)
    ax.scatter(X0, X1, c=y, cmap=cmap, edgecolors='k')
    return ax

def plot_svc_decision_function(X, y, model, ax=None, cmap='rainbow', plot_support=True):
    if ax is None:
        fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    ax.scatter(X0, X1, c=y, cmap=cmap, edgecolors='k')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


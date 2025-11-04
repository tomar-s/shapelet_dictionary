from math import log
from scipy.spatial import KDTree
import numpy as np

def estimate(X, Y, k=3, rounding=5):
    # 1. seems better to use slightly larger k for high dimensional data
    # 2. choice of rounding seems ok, but probably needs double checked
    X = np.asarray(X)
    Y = np.asarray(Y)
    #print(X.shape)
    d = X.shape[1]
    n = X.shape[0]
    m = Y.shape[0]
    #print(d,m,n)

    if rounding == None:
        X_nodups, X_counts = np.unique(X, axis=0, return_counts=True)
        Y_nodups, Y_counts = np.unique(Y, axis=0, return_counts=True)
    else:
        # use round() to crudely merge nearby points
        X_nodups, X_counts = np.unique(X.round(decimals=rounding), axis=0, return_counts=True)
        Y_nodups, Y_counts = np.unique(Y.round(decimals=rounding), axis=0, return_counts=True)
    X_tree = KDTree(X_nodups)
    Y_tree = KDTree(Y_nodups)

    k_p=[]; k_q=[]; rho=[]; nu=[]
    for x in X_nodups:
        dist,indices=X_tree.query(x, k+1) # k+1 coz count includes point itself
        k_p.append(np.sum(X_counts[indices])) # number of points in ball, incl duplicates
        rho.append(np.max(dist)) # max distance

        dist,indices=Y_tree.query(x, k)
        k_q.append(np.sum(Y_counts[indices]))
        nu.append(np.max(dist))  # max distance
    r=0
    # use the following sums to account for overlaps between the kNN sets
    # of different points
    kp_sum=sum(k_p); kq_sum=sum(k_q)
    for i in range(len(X_nodups)):
        logp = log(k_p[i]/kp_sum)-d*log(rho[i])
        logq = log(k_q[i]/kq_sum)-d*log(nu[i])
        r=r+k_p[i]/kp_sum*( logp-logq )
    return r

shapelets_jog_treadmill = np.loadtxt(
    '/scripts/deep_analysis/full_pipeline_shapelets_unnormalised_batch_wip_run2/jog_treadmill_shapelets_10.csv')
shapelets_walk_mixed = np.loadtxt(
    '/scripts/deep_analysis/full_pipeline_shapelets_unnormalised_batch_wip_run2/walk_mixed_shapelets_10.csv')
r = estimate(shapelets_walk_mixed, shapelets_jog_treadmill, k=2, rounding=None)
print(r)
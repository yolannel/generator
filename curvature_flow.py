'''
Implementation of 'AMSS' using the implementation from here: https://github.com/nilx/megawave/blob/d3cd6b8152d4933b563ac224a123583335180436/src/image/filter/amss.c
'''
import numpy as np
from numba import jit, prange
# import torch as pt
from tqdm import tqdm
import os


@jit(nopython=True, boundscheck=False, cache=True)
def pointwiseFancyRHS(u, threshold):
    ''' u.shape = (3,3), threshold = float
    returns |Du| k(u)^1/3 = (u_y^2 u_xx - 2u_x u_y u_xy + u_x^2 u_yy) ^ (1/3) 
    Note 0.353553391 = 1/sqrt(8)
    '''
    u_x = u[2, 1] - u[0, 1] + (0.353553391) * (u[2, 2] + u[2, 0] - u[0, 2] - u[0, 0])
    u_y = u[1, 2] - u[1, 0] + (0.353553391) * (u[2, 2] + u[0, 2] - u[2, 0] - u[0, 0])

    u_x, u_xy, u_y = u_x**2, u_x * u_y, u_y**2
    norm2 = u_x + u_y

    if norm2 < threshold:
        return 0.0
    else:
        u_x /= norm2; u_xy /= norm2; u_y /= norm2
        L0 = 0.5 - u_xy**2
        GK = (
            (-4 * L0) * u[1, 1]
            + (2 * L0 - u_x) * (u[0, 1] + u[2, 1])
            + (2 * L0 - u_y) * (u[1, 0] + u[1, 2])
            + (-L0 + .5 * (u_x + u_y - u_xy)) * (u[2, 2] + u[0, 0])  # u_x+u_y=1?
            + (-L0 + .5 * (u_x + u_y + u_xy)) * (u[0, 2] + u[2, 0])  # u_x+u_y=1?
        )
        # Apparently |Du|k^1/3 = (|Du|^2GK)^1/3
        return np.cbrt(GK * (norm2 * 0.0857864376))  # 0.0857864376 = (2+sqrt(2))^-2


@jit(nopython=True, boundscheck=False, cache=True, parallel=True)
def doFancystep(un, unp1, dt):
    threshold = 1e-10
    for i in prange(un.shape[0]):
        im, ip = int(max(i - 1, 0)), int(min(i + 1, un.shape[0] - 1))
        buf = np.empty((3, 3), dtype='float32')

        j = 0
        buf[0, 0], buf[0, 1], buf[0, 2] = un[im, 0], un[im, 0], un[im, 1]
        buf[1, 0], buf[1, 1], buf[1, 2] = un[i, 0], un[i, 0], un[i, 1]
        buf[2, 0], buf[2, 1], buf[2, 2] = un[ip, 0], un[ip, 0], un[ip, 1]
        unp1[i, j] = un[i, j] + dt * pointwiseFancyRHS(buf, threshold)

        j += 1
        while j < un.shape[1] - 1:
            buf[:, :2] = buf[:, 1:]
            buf[0, 2], buf[1, 2], buf[2, 2] = un[im, j + 1], un[i, j + 1], un[ip, j + 1]
            # buf[0, 0], buf[0, 1], buf[0, 2] = un[im, j - 1], un[im, j], un[im, j + 1]
            # buf[1, 0], buf[1, 1], buf[1, 2] = un[i, j - 1], un[i, j], un[i, j + 1]
            # buf[2, 0], buf[2, 1], buf[2, 2] = un[ip, j - 1], un[ip, j], un[ip, j + 1]
            unp1[i, j] = un[i, j] + dt * pointwiseFancyRHS(buf, threshold)
            j += 1

        buf[:, :2] = buf[:, 1:]  # buf[:, 2] stays the same
        unp1[i, j] = un[i, j] + dt * pointwiseFancyRHS(buf, threshold)

@jit(nopython=True, boundscheck=False, cache=True, parallel=True)
def doFancystepRGB(un, unp1, dt):
    threshold = 1e-10
    for i in prange(un.shape[0]):
        im, ip = int(max(i - 1, 0)), int(min(i + 1, un.shape[0] - 1))
        buf = np.empty((3, 3, 3), dtype='float32')

        j = 0
        buf[0, 0], buf[0, 1], buf[0, 2] = un[im, 0], un[im, 0], un[im, 1]
        buf[1, 0], buf[1, 1], buf[1, 2] = un[i, 0], un[i, 0], un[i, 1]
        buf[2, 0], buf[2, 1], buf[2, 2] = un[ip, 0], un[ip, 0], un[ip, 1]
        for k in range(3):
            unp1[i, j, k] = un[i, j, k] + dt * pointwiseFancyRHS(buf[:,:,k], threshold)

        j += 1
        while j < un.shape[1] - 1:
            buf[:, :2] = buf[:, 1:]
            buf[0, 2], buf[1, 2], buf[2, 2] = un[im, j + 1], un[i, j + 1], un[ip, j + 1]
            for k in range(3):
                unp1[i, j, k] = un[i, j, k] + dt * pointwiseFancyRHS(buf[:,:,k], threshold)
            j += 1

        buf[:, :2] = buf[:, 1:]  # buf[:, 2] stays the same
        for k in range(3):
            unp1[i, j, k] = un[i, j, k] + dt * pointwiseFancyRHS(buf[:,:,k], threshold)


@jit(nopython=True, boundscheck=False, cache=True)
def pointwiseLazyRHS(u, threshold):
    ''' u.shape = (3,3), threshold = float
    returns |Du| k(u)^1/3 = (u_y^2 u_xx - 2u_x u_y u_xy + u_x^2 u_yy) ^ (1/3) 
    '''
    u_x = .5 * (u[2, 1] - u[0, 1])
    u_y = .5 * (u[1, 2] - u[1, 0])
    u_xx = u[2, 1] - 2 * u[1, 1] + u[0, 1]
    u_yy = u[1, 2] - 2 * u[1, 1] + u[1, 0]
    u_xy = .25 * (u[2, 2] - u[2, 0] - u[0, 2] + u[0, 0])

    return np.cbrt(u_y**2 * u_xx - 2 * u_x * u_y * u_xy + u_x**2 * u_yy)


@jit(nopython=True, boundscheck=False, cache=True, parallel=True)
def doLazystep(un, unp1, dt):
    threshold = 1e-10
    for i in prange(un.shape[0]):
        im, ip = int(max(i - 1, 0)), int(min(i + 1, un.shape[0] - 1))
        buf = np.empty((3, 3), dtype='float32')

        j = 0
        buf[0, 0], buf[0, 1], buf[0, 2] = un[im, 0], un[im, 0], un[im, 1]
        buf[1, 0], buf[1, 1], buf[1, 2] = un[i, 0], un[i, 0], un[i, 1]
        buf[2, 0], buf[2, 1], buf[2, 2] = un[ip, 0], un[ip, 0], un[ip, 1]
        unp1[i, j] = un[i, j] + dt * pointwiseLazyRHS(buf, threshold)

        j += 1
        while j < un.shape[1] - 1:
            buf[:, :2] = buf[:, 1:]
            buf[0, 2], buf[1, 2], buf[2, 2] = un[im, j + 1], un[i, j + 1], un[ip, j + 1]
            # buf[0, 0], buf[0, 1], buf[0, 2] = un[im, j - 1], un[im, j], un[im, j + 1]
            # buf[1, 0], buf[1, 1], buf[1, 2] = un[i, j - 1], un[i, j], un[i, j + 1]
            # buf[2, 0], buf[2, 1], buf[2, 2] = un[ip, j - 1], un[ip, j], un[ip, j + 1]
            unp1[i, j] = un[i, j] + dt * pointwiseLazyRHS(buf, threshold)
            j += 1

        buf[:, :2] = buf[:, 1:]  # buf[:, 2] stays the same
        unp1[i, j] = un[i, j] + dt * pointwiseLazyRHS(buf, threshold)

@jit(nopython=True, boundscheck=False, cache=True, parallel=True)
def doLazystepRGB(un, unp1, dt):
    threshold = 1e-10
    for i in prange(un.shape[0]):
        im, ip = int(max(i - 1, 0)), int(min(i + 1, un.shape[0] - 1))
        buf = np.empty((3, 3, 3), dtype='float32')

        j = 0
        buf[0, 0], buf[0, 1], buf[0, 2] = un[im, 0], un[im, 0], un[im, 1]
        buf[1, 0], buf[1, 1], buf[1, 2] = un[i, 0], un[i, 0], un[i, 1]
        buf[2, 0], buf[2, 1], buf[2, 2] = un[ip, 0], un[ip, 0], un[ip, 1]
        for k in range(3):
            unp1[i, j, k] = un[i, j, k] + dt * pointwiseLazyRHS(buf[:,:,k], threshold)

        j += 1
        while j < un.shape[1] - 1:
            buf[:, :2] = buf[:, 1:]
            buf[0, 2], buf[1, 2], buf[2, 2] = un[im, j + 1], un[i, j + 1], un[ip, j + 1]
            for k in range(3):
                unp1[i, j, k] = un[i, j, k] + dt * pointwiseLazyRHS(buf[:,:,k], threshold)
            j += 1

        buf[:, :2] = buf[:, 1:]  # buf[:, 2] stays the same
        for k in range(3):
            unp1[i, j, k] = un[i, j, k] + dt * pointwiseLazyRHS(buf[:,:,k], threshold)

def computeFlow(u, dx, dt, Tmax, frames, version='fancy'):
    t = 0
    U = [u[None].copy()]
    u, v = u.copy(), u.copy()
    frame, frames = 1, np.linspace(0, Tmax, frames)

    if u.ndim==2:
        isRGB = False
    elif u.ndim==3 and u.shape[-1]==3:
        isRGB = True
    else:
        raise NotImplementedError

    if version=='fancy':
        step = doFancystepRGB if isRGB else doFancystep
    else:
        step = doLazystepRGB if isRGB else doLazystep

    # lazy hack of what I want, T[-1] >= Tmax
    T = np.arange(dt, Tmax + 3 * dt, dt)
    while T[-2] >= Tmax:
        T = T[:-1]

    for t in tqdm(T):
        step(u, v, dt / dx**(4 / 3))
        if t >= frames[frame]:
            U.append(v[None].copy())
            frame += 1
        u, v = v, u
    return u, frames, np.concatenate(U, axis=0)

def curvature_pde_np(X, u):
    # returns u_t^3 - |Du|^3 k(u) = u_t^3 - (u_y^2 u_xx - 2u_x u_y u_xy + u_x^2 u_yy)
    if X.shape[0] == 0:
        return 0.0
    U = u(X)
    
    # Compute first derivatives
    dU_t = np.gradient(U)[0][..., None]
    dU_x, dU_y = np.gradient(U)[1:]
    dU_x, dU_y = dU_x[..., None], dU_y[..., None]
    
    # Compute second derivatives
    dU_x2 = np.gradient(dU_x[..., 0])[1][..., None]
    dU_xy = np.gradient(dU_y[..., 0])[1][..., None]
    dU_y2 = np.gradient(dU_y[..., 0], axis=0)[0][..., None]
    
    return dU_t**3 - (dU_y**2 * dU_x2 - 2 * dU_x * dU_y * dU_xy + dU_x**2 * dU_y2)

# def curvature_pde(X, u):  # X[i] = (t, x, y)
#     # returns u_t^3 - |Du|^3 k(u) = u_t^3 - (u_y^2 u_xx - 2u_x u_y u_xy + u_x^2 u_yy)
#     if X.shape[0] == 0:
#         return pt.tensor(0.0)
#     U = u(X)
#     dU = pt.autograd.grad(U, X,  # first derivative
#                           grad_outputs=pt.ones_like(U),
#                           retain_graph=True,
#                           create_graph=True)[0]
#     U_t, U_x, U_y = dU[:, [0]], dU[:, [1]], dU[:, [2]]
#     dU_x = pt.autograd.grad(U_x, X,  # second derivative
#                             grad_outputs=pt.ones_like(U_x),
#                             retain_graph=True,
#                             create_graph=True)[0]
#     U_xx, U_xy = dU_x[:, [1]], dU_x[:, [2]]
#     U_yy = pt.autograd.grad(U_y, X,  # second derivative
#                             grad_outputs=pt.ones_like(U_y),
#                             retain_graph=True,
#                             create_graph=True)[0][:, [2]]
#     return U_t**3 - (U_y**2 * U_xx - 2 * U_x * U_y * U_xy + U_x**2 * U_yy)
#     # return U_t - pt.pow(U_y**2 * U_xx - 2 * U_x * U_y * U_xy + U_x**2 * U_yy, 1/3)



# def PINNrun(u, dx, Tmax, frames, interior=1e5, layers=[100] * 4, maxiter=(5e3, 1e4), load=None,
#             loss=('MSE', .1, 'MSE'), threshold=1e-3, verbosity=2, print_freq=1000, Adam={}, LBFGS={}):
#     # Decide where to save if necessary
#     if load is None:
#         load = False
#     else:
#         load = (load,) if type(load) is str else load
#         load = os.path.join(*load)
#         os.makedirs(os.path.dirname(load), exist_ok=True)

#     # Top corner of bounding box of domain
#     sz = u.shape
#     box = (Tmax, (sz[0] - 1) * dx, (sz[1] - 1) * dx)

#     # Choose boundary points:
#     S = np.mgrid[:1, :sz[0], :sz[1]]  # S = slice at time=0
#     S = np.concatenate([x.reshape(-1, 1) for x in S], axis=1) * dx
#     ####
#     # boundary = np.logical_or(np.logical_or(S[:, 1] == 0, S[:, 1] == box[1]),
#     #                          np.logical_or(S[:, 2] == 0, S[:, 2] == box[2]))
#     # boundary = S[boundary, :]  # = [0,x,y] for all boundary pixels (x,y)
#     # X = np.concatenate([S] +  # initial slice
#     #                    [boundary + np.array([[t, 0, 0]]) for t in np.linspace(0, box[0], max(sz))[1:]], axis=0)
#     # u = np.pad(u.ravel(), ((0, X.shape[0] - u.size), ))
#     ##
#     X, u = S, u.ravel()  # only fit t=0 slice
#     X, u = X[::9], u[::9]
#     ####

#     # Choose random interior points:
#     interior = np.random.rand(int(interior), X.shape[1]) * np.array([box])

#     layers = [X.shape[1]] + list(layers) + [1]
#     model = PINN_model(X, u[:, None], interior, curvature_pde, DenseNN(layers, activation='relu'), loss=loss)

#     if load and os.path.exists(load):
#         model.load(load)
#     else:
#         if maxiter[0] > 0:
#             params = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                           weight_decay=0, amsgrad=False)
#             params.update(Adam)
#             model.compile('Adam', **params)
#             model.train(iters=int(maxiter[0]), threshold=threshold, verbosity=verbosity, print_freq=print_freq)
#         if maxiter[1] > 0:
#             params = dict(lr=1.0, max_iter=1000, max_eval=10000, history_size=50,
#                           tolerance_grad=1e-8, tolerance_change=0, line_search_fn='strong_wolfe')
#             params.update(LBFGS)
#             model.compile('LBFGS')
#             model.train(iters=int(maxiter[1]), verbosity=verbosity)
#     frames = np.linspace(0, box[0], frames)

#     if load and not os.path.exists(load):
#         model.save(load)
#     return model, frames, np.concatenate([model.predict(S + np.array([[t, 0, 0]])).reshape(1, *sz) for t in frames], axis=0)

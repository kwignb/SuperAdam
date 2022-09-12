from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import neural_tangents as nt


def flatten_lg(gx):
    return jnp.reshape(gx, (-1,))


def unflatten_lg(gx, output_dimension):
    return jnp.reshape(gx, (-1, output_dimension))


def flatten_features(kernel):
    
    if kernel.ndim == 2:
        return kernel
    assert kernel.ndim % 2 == 0
    
    half_shape = (kernel.ndim - 1) // 2
    n1, n2 = kernel.shape[:2]
    feature_size = int(jnp.prod(jnp.array(kernel.shape[2 + half_shape:])))
    transposition = ((0,) + tuple(i + 2 for i in range(half_shape)) +
                    (1,) + tuple(i + 2 + half_shape for i in range(half_shape)))
    kernel = jnp.transpose(kernel, transposition)
    
    return jnp.reshape(kernel, (feature_size * n1, feature_size * n2))


def solve(mat, vec, sym_pos=False):
    mat = flatten_features(mat)
    return jsp.linalg.solve(mat, vec, sym_pos=sym_pos)


def _add_damping(covariance, damping=0., diag_reg=0.):
    
    dimension = covariance.shape[0]
    if damping > 0:
        covariance += damping * jnp.eye(dimension)
    
    if diag_reg > 0:
        reg = jnp.trace(covariance) / dimension
        covariance += diag_reg * reg * jnp.eye(dimension)
    
    return covariance + damping * jnp.eye(dimension)


def solve_w_damping(mat, vec, damping, diag_reg):
    mat = flatten_features(mat)
    mat = _add_damping(mat, damping, diag_reg)
    return jsp.linalg.solve(mat, vec, assume_a='pos')


def get_solver_by_damping_value(damping, diag_reg=0.):
    if damping == 0 and diag_reg == 0:
        return solve
    return partial(solve_w_damping, damping=damping, diag_reg=diag_reg)


def _vjp(vec, f, params, inputs):
    fx = partial(f, inputs=inputs)
    _, f_vjp = jax.vjp(fx, params)
    return f_vjp(vec)[0]


def natural_gradient_mse_fn(f, output_dimension, damping, diag_reg, 
                            kernel_batch_size, device_count, store_on_device):
    
    ntk_fn = nt.batch(
        nt.empirical_ntk_fn(f, trace_axes=()),
        kernel_batch_size, device_count, store_on_device
        )
    
    _solve_w_damping = get_solver_by_damping_value(damping, diag_reg)
    
    def natural_gradient_fn(params, x, y):
        g_dd = ntk_fn(x, None, params)
        gx = flatten_lg(f(params, x) - y)
        vec = unflatten_lg(_solve_w_damping(g_dd, gx), output_dimension)
        return _vjp(vec, f, params, x)
    
    return natural_gradient_fn


def empirical_natural_gradient_fn(f):
    
    def fisher_vjp(f_, params, x):
        _, R_z = jax.jvp(f_, (params,), (x,))
        _, f_vjp = jax.vjp(f_, params)
        return f_vjp(R_z)[0]

    def gradient_fn(params, x, y):
        loss = lambda params, x, y: 0.5 * jnp.mean(
            jnp.sum((f(params, x) - y) ** 2, axis=1)
            )
        grads = jax.grad(loss)(params, x, y)
        fvp = lambda v: fisher_vjp(loss, params, v)
        grad_fn, _ = jsp.sparse.linalg.cg(fvp, grads, maxiter=10)
        return grad_fn
    
    return gradient_fn
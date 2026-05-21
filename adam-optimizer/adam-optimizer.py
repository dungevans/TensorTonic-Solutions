import numpy as np 
def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    m_t = beta1 * m + (1-beta1)*grad
    v_t = beta2*v+(1-beta2)*(grad**2)
    m_t_hat = m_t/(1-beta1**t)
    v_t_hat = v_t/(1-beta2**t)
    param_t = param-lr * (m_t_hat/(np.sqrt (v_t_hat)+eps))
    return (param_t, m_t, v_t)
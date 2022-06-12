batchsize = 5
stepsize = 1000
nRet = 500
noise = 0.1
respOpt = [1, 4, 1, 4, 8]

def generate_tmvn(batch_size=1, steps=10, size=1000, mu=np.zeros(3), cov=np.eye(3)):
    # Truncation logic / Truncated MVN instead of MVN
    i = 0
    while i < steps:
        i += 1
        batch = np.random.multivariate_normal(mean=mu, cov=cov, size=(batch_size, size))
        batch_tf = tf.convert_to_tensor(batch)  # dims: (batch_size, size, 3)
        yield (batch_tf, tf.constant([1]))  # (data, label)



def generate_m3(batchsize, stepsize, steps, noise, respOpt, nRet):
    i = 0
    while i < steps:
        i += 1

        # Sample Hyper Parameter
        mu_c = np.random.uniform(1, 30, 1)
        sig_c = np.random.uniform(0.125, 0.5) * mu_c

        mu_a = np.random.uniform(0, 0.5, 1) * mu_c
        sig_a = np.random.uniform(0.125, 0.5) * mu_a

        mu_f = np.random.uniform(0, 1, 1)
        log_mu_f = sp.special.logit(mu_f)
        sig_f = np.random.uniform(0.1, 0.3, 1) * mu_f

        D = np.eye(4)
        mu = np.array([mu_c[0], mu_a[0], log_mu_f[0], noise])
        sig = np.diag([sig_c[0] ** 2, sig_a[0] ** 2, sig_f[0] ** 2, 0.001 ** 2])
        lower = np.array([0, 0, -10e100, 0])
        upper = np.array([10e100, 10e100, 10e100, 10e100])

        # Sample Parameters
        theta = np.zeros((batchsize, stepsize, 4), dtype=float)
        for i in range(0, batchsize):
            theta[i, :, :] = rtmvn(stepsize, mu, sig, D, lower, upper, burn=100, thin=1)

        # Compute Activations
        acts = np.zeros((batchsize, stepsize, 5), dtype=float)

        for i in range(0, batchsize):
            acts[i, :, 0] = noise + theta[i, :, 0] + theta[i, :, 1]
            acts[i, :, 1] = noise + theta[i, :, 1]
            acts[i, :, 2] = noise + sp.special.expit(theta[i, :, 2]) * (theta[i, :, 0] + theta[i, :, 1])
            acts[i, :, 3] = noise + sp.special.expit(theta[i, :, 2]) * theta[i, :, 1]
            acts[i, :, 4] = np.round_(theta[i, :, 3], 2)

        # acts[i, :, :] = np.column_stack((A_IIP, A_IOP, A_DIP, A_DIOP, A_NPL))

        # Compute Summed Acts and probabilites
        def norm_acts(acts, respOpts):
            sum_acts = sum(np.transpose(respOpts * np.transpose(acts)))
            p = np.transpose(respOpts * np.transpose(acts)) / sum_acts
            return p

        p = np.zeros((batchsize, stepsize, 5), dtype=float)
        for i in range(0, batchsize):
            p[i, :, :] = np.apply_along_axis(norm_acts, 1, acts[i, :, :], respOpt)

        # Simulate Multinomial Data
        data = np.zeros((batchsize, stepsize, 5), dtype=int)

        for batch in range(0, batchsize):
            for stepsize in range(0, stepsize):
                data[batch, stepsize, :] = np.random.multinomial(nRet, p[batch, stepsize, :])

    # Merge Data and Theta

    batch = #add sampled reference table theta and data
    batch_tf = tf.convert_to_tensor(batch)  # dims: (batch_size, size, 3) # convert to tensor
    yield (batch_tf, tf.constant([1]))  # (data, label)

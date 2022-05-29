import os
import math
from numbers import Number
from numpy.lib.function_base import average
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import lib.dist as dist
import lib.flows as flows
from lib.utils import logsumexp as logsumexp


def estimate_entropies(qz_samples, qz_params, q_dist):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1~N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1~N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, S) Variable
        qz_params  (N, K, nparams) Variable
    """

    # Only take a sample subset of the samples
    qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:10000].cuda()))

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    marginal_entropies = torch.zeros(K).cuda()
    joint_entropy = torch.zeros(1).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).data).sum(1)
        # computes - log q(z) summed over minibatch
        logqz = logqz_i.sum(1)  # (N, S)
        joint_entropy += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False).data).sum(0)
        pbar.update(batch_size)
    pbar.close()

    marginal_entropies /= S
    joint_entropy /= S

    return marginal_entropies, joint_entropy


def analytical_NLL(qz_params, q_dist, prior_dist, qz_samples=None):
    """Computes the quantities
        1/N sum_n=1~N E_{q(z|x)} [ - log q(z|x) ]
    and
        1/N sum_n=1~N E_{q(z_j|x)} [ - log p(z_j) ]

    Inputs:
    -------
        qz_params  (N, K, nparams) Variable

    Returns:
    --------
        nlogqz_condx (K,) Variable
        nlogpz (K,) Variable
    """
    pz_params = Variable(torch.zeros(1).type_as(qz_params.data).expand(qz_params.size()), volatile=True)

    nlogqz_condx = q_dist.NLL(qz_params).mean(0)
    nlogpz = prior_dist.NLL(pz_params, qz_params).mean(0)
    return nlogqz_condx, nlogpz


def elbo_decomposition(vae, dataset_loader):
    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    S = 1                            # number of latent variable samples
    nparams = 2
    x_dist = dist.Bernoulli()
    q_dist = dist.Normal()
    prior_dist = dist.Normal()
    prior_params = torch.zeros(vae.z_dim, 2)

    print('Computing q(z|x) distributions.')
    # compute the marginal q(z_j|x_n) distributions
    qz_params = torch.Tensor(N, K, nparams)
    n = 0
    logpx = 0
    for xs in tqdm(dataset_loader):
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, -1, 64, 64).cuda())
        z_params = vae._encode(xs).view(batch_size, nparams, K).transpose(1,2)
        qz_params[n:n + batch_size] = z_params.data
        n += batch_size

        # estimate reconstruction term
        for _ in range(S):
            z = q_dist.sample(params=z_params)
            x_params = vae._decode(z)
            logpx += x_dist.log_density(xs, params=x_params).view(batch_size, -1).data.sum()
    # Reconstruction term
    logpx = logpx / (N * S)

    qz_params = Variable(qz_params.cuda())

    print('Sampling from q(z).')
    # sample S times from each marginal q(z_j|x_n)
    qz_params_expanded = qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)
    qz_samples = q_dist.sample(params=qz_params_expanded)
    qz_samples = qz_samples.transpose(0, 1).contiguous().view(K, N * S)

    print('Estimating entropies.')
    marginal_entropies, joint_entropy = estimate_entropies(qz_samples, qz_params, q_dist)

    if hasattr(q_dist, 'NLL'):
        nlogqz_condx = q_dist.NLL(qz_params).mean(0)
    else:
        nlogqz_condx = - q_dist.log_density(qz_samples,
            qz_params_expanded.transpose(0, 1).contiguous().view(K, N * S)).mean(1)

    if hasattr(prior_dist, 'NLL'):
        expanded_size = (N * K,) + prior_params.size()
        prior_params = Variable(prior_params.expand(expanded_size))
        pz_params = prior_params.contiguous().view(N, K, -1).cuda()
        nlogpz = prior_dist.NLL(pz_params, qz_params).mean(0)
    else:
        nlogpz = - prior_dist.log_density(qz_samples.transpose(0, 1)).mean(0)

    # nlogqz_condx, nlogpz = analytical_NLL(qz_params, q_dist, prior_dist)
    nlogqz_condx = nlogqz_condx.data
    nlogpz = nlogpz.data

    # Independence term
    # KL(q(z)||prod_j q(z_j)) = log q(z) - sum_j log q(z_j)
    dependence = (- joint_entropy + marginal_entropies.sum())[0]

    # Information term
    # KL(q(z|x)||q(z)) = log q(z|x) - log q(z)
    information = (- nlogqz_condx.sum() + joint_entropy)[0]

    # Dimension-wise KL term
    # sum_j KL(q(z_j)||p(z_j)) = sum_j (log q(z_j) - log p(z_j))
    dimwise_kl = (- marginal_entropies + nlogpz).sum()

    # Compute sum of terms analytically
    # KL(q(z|x)||p(z)) = log q(z|x) - log p(z)
    analytical_cond_kl = (- nlogqz_condx + nlogpz).sum()

    # print('Dependence: {}'.format(dependence))
    # print('Information: {}'.format(information))
    # print('Dimension-wise KL: {}'.format(dimwise_kl))
    # print('Analytical E_p(x)[ KL(q(z|x)||p(z)) ]: {}'.format(analytical_cond_kl))
    # print('Estimated  ELBO: {}'.format(logpx - analytical_cond_kl))

    return logpx, analytical_cond_kl, dependence, information, dimwise_kl, marginal_entropies, joint_entropy


import numpy as np
import matplotlib.pyplot as plt

def normal_logpdf(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x-mu)/sigma)**2

def stable_normalize_from_log(logf, x):
    a = np.max(logf)
    f = np.exp(logf - a)
    Z = np.trapz(f, x)
    return f / Z

def log_trapz_exp(logf, x):
    a = np.max(logf)
    return np.log(np.trapz(np.exp(logf - a), x)) + a

def subset_iter(K):
    for mask in range(1<<K):
        inds = [i for i in range(K) if (mask>>i)&1]
        yield mask, inds

def mask_to_label(mask, K):
    if mask == 0:
        return "∅"
    return "{" + ",".join(str(i+1) for i in range(K) if (mask>>i)&1) + "}"

# Grid and prior
theta = np.linspace(-6.0, 6.0, 6001)
mu0, s0 = 0.0, 1.5
log_pi = normal_logpdf(theta, mu0, s0)
pi = np.exp(log_pi)

# Simulate experiments
theta_true = 1.0
sigmas = np.array([0.35, 0.35, 0.40])
y = np.array([1.05, theta_true + 3.0, theta_true - 4.0])

# Likelihoods and posteriors
log_L = np.vstack([normal_logpdf(y[i], theta, sigmas[i]) for i in range(3)])
L = np.exp(log_L)
log_p = log_pi + log_L
p = np.vstack([stable_normalize_from_log(log_p[i], theta) for i in range(3)])

# Collapsed posterior with independent Bernoulli priors on validity
rho = np.array([0.5, 0.5, 0.5])
log_terms = np.log(1.0 - rho)[:, None] + np.log1p((rho/(1.0 - rho))[:, None] * np.exp(log_L))
log_coll = log_pi + np.sum(log_terms, axis=0)
p_coll = stable_normalize_from_log(log_coll, theta)

# Subset posteriors and weights (Bernoulli prior)
subset_posteriors = {}
subset_labels = []
subset_logw = []
for mask, inds in subset_iter(3):
    log_pS_unnorm = log_pi + (np.sum(log_L[inds], axis=0) if inds else 0.0)
    log_mS = log_trapz_exp(log_pS_unnorm, theta)  # evidence
    pS = np.exp(log_pS_unnorm - log_mS)
    subset_posteriors[mask] = pS
    z = np.array([(mask>>i)&1 for i in range(3)])
    log_prior_mass = (z*np.log(rho) + (1-z)*np.log(1-rho)).sum()
    subset_logw.append(log_prior_mass + log_mS)
    subset_labels.append(mask_to_label(mask, 3))
subset_logw = np.array(subset_logw)
w = np.exp(subset_logw - np.max(subset_logw))
w /= w.sum()
p_mix = np.zeros_like(theta)
for weight, (mask, _) in zip(w, subset_iter(3)):
    p_mix += weight * subset_posteriors[mask]

# Exactly-one-valid alternative prior
def evidence_singleton(i):
    return np.exp(log_trapz_exp(log_pi + log_L[i], theta))
m_singletons = np.array([evidence_singleton(i) for i in range(3)])
w_one = (m_singletons / m_singletons.sum())
p_mix_one = (w_one[0]*subset_posteriors[1] + 
             w_one[1]*subset_posteriors[2] + 
             w_one[2]*subset_posteriors[4])

# --- Plots ---
plt.figure(figsize=(8,5))
plt.plot(theta, pi, label="Prior π(θ)")
for i in range(3):
    plt.plot(theta, p[i], label=f"Posterior p{i+1}(θ)")
plt.title("Prior and individual experiment posteriors")
plt.xlabel("θ"); plt.ylabel("density"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
for mask, _ in subset_iter(3):
    plt.plot(theta, subset_posteriors[mask], label=f"p_{mask_to_label(mask,3)}(θ)")
plt.title("Subset posteriors p_S(θ)")
plt.xlabel("θ"); plt.ylabel("density"); plt.legend(ncol=2, fontsize=9); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(theta, p_coll, label="Collapsed posterior (Bernoulli prior)")
plt.plot(theta, p_mix, linestyle="--", label="Mixture over subsets (Bernoulli)")
plt.title("Collapsed posterior equals subset mixture (Bernoulli prior)")
plt.xlabel("θ"); plt.ylabel("density"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.bar(subset_labels, w)
plt.title("Posterior weights w_S (Bernoulli prior)")
plt.xlabel("Subset S"); plt.ylabel("Weight"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(theta, subset_posteriors[1], label="p_{1}(θ)")
plt.plot(theta, subset_posteriors[2], label="p_{2}(θ)")
plt.plot(theta, subset_posteriors[4], label="p_{3}(θ)")
plt.plot(theta, p_mix_one, linestyle="--", label="Mixture (exactly one valid)")
plt.title("Exactly-one-valid prior: mixture over singletons")
plt.xlabel("θ"); plt.ylabel("density"); plt.legend(); plt.tight_layout(); plt.show()

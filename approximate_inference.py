import scipy.io
vars = scipy.io.loadmat('tennis_data.mat')
player_names = vars['W']
tennis_games = vars['G']
num_players = len(player_names)
print('Loaded data for', num_players, 'players')


import autograd.numpy as np
np.random.seed(414)
import matplotlib.pyplot as plt
%matplotlib inline

def log1pexp(x): # log1pexp is not available in numpy; 
                        #this is equivalent to the function defined in StatsFuns.jl
    if np.min(x) < 9:
        return np.log1p(np.exp(x))
    elif np.min(x) < 16:
        return x + np.exp(-x)
    else:
        return x

def logp_a_beats_b(za,zb):
    ll=-log1pexp(-(za-zb))
    return ll

def all_games_log_likelihood(zs,games):
    games=games-1 # python indexing starts from 0
    zs_a=zs[games[:,0]]
    zs_b=zs[games[:,1]]
    likelihoods=logp_a_beats_b(zs_a,zs_b)
    return likelihoods

def joint_log_density(zs,games):
    return log_prior(zs)+all_games_log_likelihood(zs,games)

# Convenience function for producing toy games between two players.
def two_player_toy_games(p1_wins, p2_wins):
    p1=np.array([1,2])
    p2=np.array([2,1])
    return np.vstack((np.repeat(p1[np.newaxis,:], p1_wins,0), 
                      np.repeat(p2[np.newaxis,:], p2_wins,0)))

def factorized_gaussian_log_density(mu,logsig,xs):
    sig = np.exp(logsig)
    return np.sum(-0.5*np.log(2*np.pi*sig**2)-0.5*((xs.T-mu)**2)/(sig**2), axis=1)


def elbo(params,logp,num_samples,num_players):
    mu=params[0]
    logsig=params[1]
    samples = mu + np.random.randn(num_samples,num_players)*np.exp(logsig)
    logp_estimate = logp(samples.T)
    logq_estimate = factorized_gaussian_log_density(mu,logsig,samples.T)
    return (logp_estimate-logq_estimate).mean()
    
# Conveinence function for taking gradients 
def neg_toy_elbo(params, games, num_samples = 100, num_players = 107):
    def logp(zs):
        return joint_log_density(zs,games)
    return -elbo(params,logp,num_samples,num_players)

init_mu = np.random.randn(num_players)/100 # Initial mu, can initialize randomly!
init_log_sigma = np.random.randn(num_players)/100 
                              # Initual log_sigma, can initialize randomly!
init_params = np.array([init_mu, init_log_sigma])


def fit_variational_dist(init_params, toy_evidence, num_itrs=200, 
                         lr= 1e-2, num_q_samples = 10,verbose=0):
    params_cur = init_params
    for i in range(num_itrs):
        from autograd import grad
        grad_params=grad(neg_toy_elbo) 
        params_cur = params_cur - lr*grad_params(params_cur,games)
        if(verbose):
            print('Iteration: ', i)
            print('Current ELBO: ',-neg_toy_elbo(params_cur,games))
    print('Final Loss: ', neg_toy_elbo(params_cur,games))
    return params_cur

trained_params=fit_variational_dist(init_params, tennis_games, 
                                    num_itrs=200, lr= 1e-2, num_q_samples = 10)

mean=trained_params[0]
variance=np.exp(trained_params[1])**2
fig, ax=plt.subplots()
perm=np.argsort(mean)
ax.plot(mean[perm], variance[perm],'b')
ax.set_xlabel('Mean')
ax.set_ylabel('Variance')
ax.set_title('Approximate mean and variance sorted by skill')
plt.show()


print('Names of the 10 players with the highest mean skill under the variational model')
print(W[perm[::-1][:10]])


mu = mean[0]+mean[4] #0: Rafael Nadal; 4: Roger Federer
logsig = np.log(np.sqrt(variance[0]**2+variance[4]**2))

def variational_posterior(zs):
    return np.exp(factorized_gaussian_log_density(mu,logsig,zs))

fig, ax = plt.subplots()
title='Joint approximate posterior over the skills of Rafael Nadal and Roger Federer'
skillcontour(variational_posterior,title,fig,ax)
plot_line_equal_skill()
plt.show()

from scipy.stats import norm
print('Probability that Roger Federer has higher skill than Rafael Nadal')
ext_p = 1-norm.cdf(0,mean[4]-mean[0],np.sqrt(variance[0]+variance[4]))
print('Exact Probability: ',ext_p)

mc_p = np.mean(mean[4]-mean[0]+np.sqrt(variance[0]
                                       +variance[4])*np.random.randn(10000)>=0)
print('Monte-Carlo: ',mc_p)

from scipy.stats import norm
print('Probability that Roger Federer has higher skill than the player with lowest skill')
ext_p = 1-norm.cdf(0,mean[4]-mean[perm[0]],np.sqrt(variance[perm[0]]+variance[4]))
print('Exact Probability: ',ext_p)

mc_p = np.mean(mean[4]-mean[perm[0]]+np.sqrt(variance[perm[0]]
                                             +variance[4])*np.random.randn(10000)>=0)
print('Monte-Carlo: ',mc_p)
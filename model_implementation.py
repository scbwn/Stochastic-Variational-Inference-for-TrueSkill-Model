import numpy as np

def log_prior(zs):
    N=zs.shape[0]
    return -0.5*N*np.log(2*np.pi)-0.5*np.sum(zs**2, axis=0)


def log1pexp(x): # log1pexp is not available in numpy; this is equivalent to the 
                                     # function defined in StatsFuns.jl
    sum=0
    for j in range(len(x)): 
        if x[j] < 9:
            sum+=np.log1p(np.exp(x[j]))
        elif x[j] < 16:
            sum+=x[j] + np.exp(-x[j])
        else:
            sum+=x[j]
    return sum

def logp_a_beats_b(za,zb):
    ll=np.zeros(za.shape[1])
    for i in range(za.shape[1]):
        ll[i]=-log1pexp(-(za[:,i]-zb[:,i]))
    return ll

def all_games_log_likelihood(zs,games):
    games=games-1 # python indexing starts from 0
    zs_a=zs[games[:,0]]
    zs_b=zs[games[:,1]]
    likelihoods=logp_a_beats_b(zs_a,zs_b)
    return likelihoods

def joint_log_density(zs,games):
    return log_prior(zs)+all_games_log_likelihood(zs,games)


# Test shapes of batches for likelihoods
np.random.seed(414)
B = 15 # number of elements in batch
N = 4 # Total Number of Players

test_zs = np.random.randn(4,15)
test_games = np.asarray([[1,2],[3,1],[4,2]]) # 1 beat 2, 3 beat 1, 4 beat 2
print('Check test_zs: ', test_zs.shape==(N,B))

#batch of priors
print('Check log_prior: ', len(log_prior(test_zs)) == B)

# loglikelihood of p1 beat p2 for first sample in batch
print('Check logp_a_beats_b: ', len(logp_a_beats_b(test_zs[1,1].reshape(1,1),
                                                   test_zs[2,1].reshape(1,1)))==1)

# loglikelihood of p1 beat p2 broadcasted over whole batch
print('Check logp_a_beats_b over whole batch: ', 
      len(logp_a_beats_b(test_zs[1,:].reshape(1,len(test_zs[1,:])),
                         test_zs[2,:].reshape(1,len(test_zs[2,:]))))==B)

# batch loglikelihood for evidence
print('Check all_games_log_likelihood: ', 
      len(all_games_log_likelihood(test_zs,test_games))==B)

# batch loglikelihood under joint of evidence and prior
print('Check joint_log_density: ', len(joint_log_density(test_zs,test_games))==B)
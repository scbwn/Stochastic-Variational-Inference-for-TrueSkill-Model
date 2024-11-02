import autograd.numpy as np
np.random.seed(414)
import matplotlib.pyplot as plt
%matplotlib inline

def log_prior(zs):
    N=zs.shape[0]
    return -0.5*N*np.log(2*np.pi)-0.5*np.sum(zs**2, axis=0)

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

def skillcontour(f, title, fig, ax, colour=None):
    n = 100
    x = np.linspace(-3,3,n)
    y = np.linspace(-3,3,n)
    X,Y = np.meshgrid(x,y) # meshgrid for contour
    z_grid = np.vstack([X.reshape(X.shape[0]*X.shape[1]),
                        Y.reshape(Y.shape[0]*Y.shape[1])]) # add single batch dim
    Z = f(z_grid)
    Z=Z.reshape(n,n)
    max_z = np.max(Z)
    if max_z<=0:
        levels = max_z/np.array([.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2])
    else: 
        levels = max_z*np.array([.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2])
    
    if colour == None:
        p1 = ax.contour(X, Y, Z, levels[::-1])
    else:
        p1 = ax.contour(X, Y, Z, levels[::-1], colors=colour)
    ax.clabel(p1, inline=1, fontsize=10)
    ax.set_xlabel('Player 1 Skill')
    ax.set_ylabel('Player 2 Skill')
    ax.set_title(title)
    return

def plot_line_equal_skill():
    x=np.linspace(-3,3,200)
    y=np.linspace(-3,3,200)
    plt.plot(x,y,'g', label='Equal Skill')
    plt.legend()
    return

def elbo(params,logp,num_samples,num_players):
    mu=params[0]
    logsig=params[1]
    samples = mu + np.random.randn(num_samples,num_players)*np.exp(logsig)
    logp_estimate = logp(samples.T)
    logq_estimate = factorized_gaussian_log_density(mu,logsig,samples.T)
    return (logp_estimate-logq_estimate).mean()
    
    
# Conveinence function for taking gradients 
def neg_toy_elbo(params, games, num_samples = 100, num_players = 2):
    def logp(zs):
        return joint_log_density(zs,games)
    return -elbo(params,logp,num_samples,num_players)

num_players_toy = 2 
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = np.array([toy_mu, toy_ls])


def fit_toy_variational_dist(init_params, games, num_itrs=200, lr= 1e-2, 
                             num_q_samples = 10,verbose=0):
    params_cur = init_params
    for i in range(num_itrs):
        from autograd import grad
        grad_params=grad(neg_toy_elbo) 
        params_cur = params_cur - lr*grad_params(params_cur,games)
        if(verbose):
            print('Iteration: ', i)
            print('Current ELBO: ',-neg_toy_elbo(params_cur,games))
        mu=params_cur[0]
        logsig=params_cur[1]
        samples = mu + np.random.randn(100,2)*np.exp(logsig)
        
        logq_estimate = factorized_gaussian_log_density(mu,logsig,samples.T)
        plt.plot(i,np.exp(logq_estimate.mean()-neg_toy_elbo(params_cur,games)),'ro')
        plt.plot(i,np.exp(logq_estimate.mean()),'bo')
    plt.title('True posterior in red and Variational in blue')
    plt.show()
    
    mu=params_cur[0]
    logsig=params_cur[1]
    samples = mu + np.random.randn(100,2)*np.exp(logsig)
    def variational_posterior(zs):
        return factorized_gaussian_log_density(mu,logsig,zs)
    def target_posterior(zs):
        return factorized_gaussian_log_density(mu,logsig,zs)-neg_toy_elbo(params_cur,games)
    
    fig, ax = plt.subplots()
    title='Target posterior in red and Variational in blue'
    skillcontour(variational_posterior,title,fig,ax,colour='b')
    skillcontour(target_posterior,title,fig,ax,colour='r')
    plot_line_equal_skill()
    plt.show()
    print('Final Loss: ', neg_toy_elbo(params_cur,games))
    return params_cur

games = two_player_toy_games(1,0)
result_1=fit_toy_variational_dist(toy_params_init, games, num_itrs=200, 
                               lr= 1e-2, num_q_samples = 10,verbose=1)
print(result_1)


games = two_player_toy_games(10,0)
result_2=fit_toy_variational_dist(toy_params_init, games, num_itrs=200, 
                               lr= 1e-2, num_q_samples = 10)
print(result_2)


games = two_player_toy_games(10,10)
result_3=fit_toy_variational_dist(toy_params_init, games, num_itrs=200, 
                               lr= 1e-2, num_q_samples = 10)
print(result_3)
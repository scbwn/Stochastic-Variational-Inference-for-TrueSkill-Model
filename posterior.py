import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def skillcontour(f, title, colour=None):
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
    
    fig, ax = plt.subplots()
    
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
    plt.plot(x,y,'r', label='Equal Skill')
    plt.legend()
    return


# Convenience function for producing toy games between two players.
def two_player_toy_games(p1_wins, p2_wins):
    p1=np.array([1,2])
    p2=np.array([2,1])
    return np.vstack((np.repeat(p1[np.newaxis,:], p1_wins,0), 
                      np.repeat(p2[np.newaxis,:], p2_wins,0)))

def log_prior(zs):
    N=zs.shape[0]
    return -0.5*N*np.log(2*np.pi)-0.5*np.sum(zs**2, axis=0)

def joint_prior(zs):
    return np.exp(log_prior(zs))

skillcontour(joint_prior, 'Joint Prior Contour Plot')
plot_line_equal_skill()


skillcontour(log_prior, 'Likelihood Function Contour Plot')
plot_line_equal_skill()

def log1pexp(x): # log1pexp is not available in numpy; this is equivalent 
                                        #to the function defined in StatsFuns.jl
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

def q2c_joint_posterior(zs):
    return joint_log_density(zs,two_player_toy_games(1,0))


skillcontour(q2c_joint_posterior, 'Joint Posterior Contour Plot (A beat B in 1 game)')
plot_line_equal_skill()


def q2d_joint_posterior(zs):
    return joint_log_density(zs,two_player_toy_games(10,0))


skillcontour(q2d_joint_posterior, 
             'Joint Posterior Contour Plot (A beat B in all 10 games)')
plot_line_equal_skill()


def q2e_joint_posterior(zs):
    return joint_log_density(zs,two_player_toy_games(10,10))


skillcontour(q2e_joint_posterior, 
             'Joint Posterior Contour Plot (A and B beat each other in 10 games each)')
plot_line_equal_skill()
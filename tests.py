#This file holds code that test our pipeline - pipeline.py
#It tests the functions in pipeline.py
#This allows me to reconstruct any test images
#Or run any tests I want to run

from pipeline import Source, Lens, score_map
from utils import process_weights, find_eR
import plots
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from multiprocessing import Pool

#Global variables
sigma_f = 10**-3 #Noise level for flexion
sigma_g = 10**-3 #Noise level for shear


def make_sources(Nsource, size = 50):
    #Create a set of sources
    #Uniformly distributed in a circle of radius size/2 - size gives the length of the side of the square
    rs = np.sqrt(np.random.random(Nsource)) * (size)
    thetas = np.random.random(Nsource) * 2*np.pi
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    return xs, ys


def noise_variance():
    #Generates the likelihood maps for a single source with different
    #estimated noise levels
    #THIS NEEDS TO BE REWRITTEN TO ACCOUNT FOR THE CHANGE TO SOURCE CLASS
    size = 50
    Nsource = 1
    line = np.linspace(-size,size,size)

    #Create lens
    xlens = [0]
    ylens = [0]
    eR = [5]
    lenses = Lens(xlens, ylens, eR)

    #Create sources
    xs, ys = make_sources(Nsource, size = size)

    sources = Source(xs, ys, np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource))
    sources.calc_shear(lenses)
    sources.calc_flex(lenses)

    #First flexion
    weights1 = sources.fweight(size,sigma=10**-1)
    weights2 = sources.fweight(size,sigma=10**-2)
    weights3 = sources.fweight(size,sigma=10**-3)

    map1 = process_weights(weights1, np.linspace(1,60,size), size)
    map2 = process_weights(weights2, np.linspace(1,60,size), size)
    map3 = process_weights(weights3, np.linspace(1,60,size), size)

    x1,y1,z1 = score_map(map1, threshold=0.5, N = 3)
    xmax1, ymax1 = line[x1], -line[y1]

    x2,y2,z2 = score_map(map2, threshold=0.5, N = 3)
    xmax2, ymax2 = line[x2], -line[y2]

    x3,y3,z3 = score_map(map3, threshold=0.5, N = 3)
    xmax3, ymax3 = line[x3], -line[y3]

    fig, ax = plt.subplots(2,3, figsize=(10,5), sharex=True, sharey=True)
    fig.suptitle('One Source: Noise Estimates', fontsize=16)

    plots.plot_likelihood_map(ax[0,0],np.log10(map1 + 10**-20),lenses,sources,xmax1,ymax1,100*z1,size,'Flexion Weighting: Overestimate Noise')
    plots.plot_likelihood_map(ax[0,1],np.log10(map2 + 10**-20),lenses,sources,xmax2,ymax2,100*z2,size,'Flexion Weighting: Correct Noise')
    plots.plot_likelihood_map(ax[0,2],np.log10(map3 + 10**-20),lenses,sources,xmax3,ymax3,100*z3,size,'Flexion Weighting: Underestimate Noise')

    #And now shear
    weights1 = sources.gweight(size,sigma=10**-1)
    weights2 = sources.gweight(size,sigma=10**-2)
    weights3 = sources.gweight(size,sigma=10**-3)

    x1,y1,z1 = score_map(weights1, threshold=0.5, N = 3)
    xmax1, ymax1 = line[x1], -line[y1]

    x2,y2,z2 = score_map(weights2, threshold=0.5, N = 3)
    xmax2, ymax2 = line[x2], -line[y2]

    x3,y3,z3 = score_map(weights3, threshold=0.5, N = 3)
    xmax3, ymax3 = line[x3], -line[y3]

    plots.plot_likelihood_map(ax[1,0],np.log10(weights1),lenses,sources,xmax1,ymax1,100*z1,size,'Shear Weighting: Overestimate Noise')
    plots.plot_likelihood_map(ax[1,1],np.log10(weights2),lenses,sources,xmax2,ymax2,100*z2,size,'Shear Weighting: Correct Noise')
    plots.plot_likelihood_map(ax[1,2],np.log10(weights3),lenses,sources,xmax3,ymax3,100*z3,size,'Shear Weighting: Underestimate Noise')

    plt.savefig('Images/noise_variance.png')


def make_lensing_field(Nlens, Nsource = 0, size = 50, Noise = True, centered = False):
    if Nsource == 0:
        Nsource = int(10**-2 * size**2)
    
    xs, ys = make_sources(Nsource, size = size)

    xl = np.random.uniform(-size,size,Nlens)
    yl = np.random.uniform(-size,size,Nlens)
    eR = np.random.uniform(0,10,Nlens)

    if centered:
        xl[0] = 0
        yl[0] = 0
        eR[0] = 5


    if Noise:
        f1 = np.random.normal(0,sigma_f,Nsource)
        f2 = np.random.normal(0,sigma_f,Nsource)
        g1 = np.random.normal(0,sigma_g,Nsource)
        g2 = np.random.normal(0,sigma_g,Nsource)
    else:
        f1 = np.zeros(Nsource)
        f2 = np.zeros(Nsource)
        g1 = np.zeros(Nsource)
        g2 = np.zeros(Nsource)

    sources = Source(xs, ys, g1, g2, f1, f2)
    lenses = Lens(xl, yl, eR)

    sources.calc_shear(lenses)
    sources.calc_flex(lenses)

    return lenses, sources


def create_test_set(Nlens, Nsource, Noise = True, centered = True, size = 50):
    #Generate a test set of lenses and sources
    #Return the weights for each weighting scheme and the coordinates of the maxima

    #Note: here 1,2,3 correspond to shear, flexion, and shear + flexion respectively
    lenses,sources = make_lensing_field(Nlens, Nsource, size = size, Noise = Noise, centered = centered)

    weights1, weights2, weights3 = sources.weights(size,sigma_f=10**-2,sigma_g=10**-2)
    weights = [weights1, weights2, weights3]

    eR_range = np.linspace(1,60,size)

    map1 = process_weights(weights1, eR_range, size)
    map2 = process_weights(weights2, eR_range, size)
    map3 = process_weights(weights3, eR_range, size)

    maps = [map1, map2, map3]

    #Find maxima within the weightmaps, output their coordinates and scores
    line = np.linspace(-size,size,size)

    x1,y1,z1 = score_map(maps[0], threshold=0.5)
    xmax1, ymax1 = line[x1], -line[y1]

    x2,y2,z2 = score_map(maps[1], threshold=0.5)
    xmax2, ymax2 = line[x2], -line[y2]

    x3,y3,z3 = score_map(maps[2], threshold=0.5)
    xmax3, ymax3 = line[x3], -line[y3]

    maxima = [[xmax1, ymax1], [xmax2, ymax2], [xmax3, ymax3]]
    strengths = [z1, z2, z3]

    #Next, find the einstein radius associated with each maxima

    eR1 = find_eR(weights[0], x1, y1, eR_range)
    eR2 = find_eR(weights[1], x2, y2, eR_range)
    eR3 = find_eR(weights[2], x3, y3, eR_range)

    eR = [eR1, eR2, eR3]

    return maps, lenses, sources, maxima, strengths, eR


def plot_test_map(Noise=False, centered = True, size = 50, Nlens = 1, Nsource = 100):
    maps, lenses, sources, maxima, strengths, eR = create_test_set(Nlens, Nsource, Noise = Noise, centered = centered, size = size)
    #Plot the results
    fig, ax = plt.subplots(1,3, figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Likelihood Maps \n Nlens = {}, Nsource = {} \n Noise = {}'.format(Nlens, Nsource, Noise), fontsize=16)
    
    plots.plot_likelihood_map(ax[0],np.log10(maps[0]+10**-20),lenses,None,maxima[0][0],maxima[0][1],100*strengths[0],eR[0],size,'Shear')
    plots.plot_likelihood_map(ax[1],np.log10(maps[1]+10**-20),lenses,None,maxima[1][0],maxima[1][1],100*strengths[1],eR[1],size,'Flexion')
    plots.plot_likelihood_map(ax[2],np.log10(maps[2]+10**-20),lenses,None,maxima[2][0],maxima[2][1],100*strengths[2],eR[2],size,'Flexion + Shear')

    #Shift the axes up a bit to remove whitespace
    plt.subplots_adjust(top=1)
    #plt.savefig('Images/tests/{}_map_lens_{}_source_{}.png'.format('noiseless' if Noise == False else 'noisy', Nlens, Nsource))
    plt.show()


def random_realization(size):
    #Create a random realization of a map
    #This is a function that can be run in parallel
    #It returns the coordinates of the maxima closest to the lens
    #For each weighting scheme
    #It also returns the coordinates of the lens
    #And the coordinates of the maxima
    #And the scores of the maxima
    #This is so we can plot the results later

    
    _, lenses, *_, maxima, strengths, eR = create_test_set(1, 0, Noise = True, centered = True, size = size)
    true_eR = lenses.eR[0]

    #Return the maxima with the highest score
    max1 = np.argmax(strengths[0])
    max2 = np.argmax(strengths[1])
    max3 = np.argmax(strengths[2])

    #Return the coordinates of the maxima
    x1 = maxima[0][0][max1]
    y1 = maxima[0][1][max1]
    x2 = maxima[1][0][max2]
    y2 = maxima[1][1][max2]
    x3 = maxima[2][0][max3]
    y3 = maxima[2][1][max3] 
    eR1 = eR[0][max1] - true_eR
    eR2 = eR[1][max2] - true_eR
    eR3 = eR[2][max3] - true_eR

    return x1, y1, eR1, x2, y2, eR2, x3, y3, eR3


def run_random_realization(Ntrials, size = 100):
    #Run the random realization function
    pbar = tqdm(total=Ntrials)

    pool = Pool(processes=20)
    results = []
    for result in pool.imap_unordered(random_realization, [size for i in range(Ntrials)]):
        results.append(result)
        pbar.update()

    #results = np.load('random_realization.npy')
    pool.close()
    
    np.save('Data/random_realization_N_{}.npy'.format(Ntrials), results)
    #Unpack the results
    x1 = np.array([result[0] for result in results])
    y1 = np.array([result[1] for result in results])
    eR1 = np.array([result[2] for result in results])
    x2 = np.array([result[3] for result in results])
    y2 = np.array([result[4] for result in results])
    eR2 = np.array([result[5] for result in results]) 
    x3 = np.array([result[6] for result in results])
    y3 = np.array([result[7] for result in results])
    eR3 = np.array([result[8] for result in results]) 

    data = [x1, y1, eR1, x2, y2, eR2, x3, y3, eR3]
    labels = ['dx (s)', 'dy (s)', r'd$\theta_E$ (s)', 'dx (f)', 'dy (f)', r'd$\theta_E$ (f)', 'dx (s + f)', 'dy (s + f)', r'd$\theta_E$ (s + f)']

    #Plot the results - we want confidence ellipses that relate (x,y), (x,eR) and (y,eR). This should be repeated for each weighting scheme
    fig, ax = plt.subplots(3,3, figsize=(15,15), sharex=True, sharey=True)
    fig.suptitle('Random Realization: {} Trials - Shear'.format(Ntrials), fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    plots.correlation_plot([x1,y1,eR1], [r'$\theta_E','y','x'], ax)
    plt.savefig('Images/rr/shear_N_{}.png'.format(Ntrials))

    fig, ax = plt.subplots(3,3, figsize=(15,15), sharex=True, sharey=True)
    fig.suptitle('Random Realization: {} Trials - Flexion'.format(Ntrials), fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    plots.correlation_plot([x2,y2,eR2], [r'$\theta_E','y','x'], ax)
    plt.savefig('Images/rr/flexion_N_{}.png'.format(Ntrials))    

    fig, ax = plt.subplots(3,3, figsize=(15,15), sharex=True, sharey=True)
    fig.suptitle('Random Realization: {} Trials - Flexion + Shear'.format(Ntrials), fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    plots.correlation_plot([x3,y3,eR3], [r'$\theta_E','y','x'], ax)
    plt.savefig('Images/rr/flexion_shear_N_{}.png'.format(Ntrials))

    #Plot correlation of all parameters
    fig, ax = plt.subplots(9,9, figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Random Realization: {} Trials'.format(Ntrials), fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    plots.correlation_plot(data, labels, ax)
    plt.savefig('Images/rr/N_{}.png'.format(Ntrials))


def run_a2744():
    #File is formatted as follows
    #ID,NAME,z,ra,dec,x ,y,xc,yc,a,f1,f2,g1,g2,xs,ys
    file = 'a2744_updated.csv'
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    x = data[:,5]
    y = data[:,6]
    f1 = data[:,10]
    f2 = data[:,11]
    g1 = data[:,12]
    g2 = data[:,13]

    #To compute the size, lets take the largest separation between two sources
    #And add 10% to it
    size = int(1.1 * np.max(np.sqrt((x[:,np.newaxis] - x)**2 + (y[:,np.newaxis] - y)**2)))

    #Create the sources
    sources = Source(x, y, g1, g2, f1, f2)

    #Compute the weights
    weights1, weights2, weights3 = sources.weights(size,sigma_f=10**-2,sigma_g=10**-3)
    #Process weights
    map1 = process_weights(weights1, np.linspace(1,60,size), size)
    map2 = process_weights(weights2, np.linspace(1,60,size), size)
    map3 = process_weights(weights3, np.linspace(1,60,size), size)
    
    np.save('Data/a2744_weights.npy', [weights1, weights2, weights3])
    #weights1, weights2, weights3 = np.load('a2744_weights.npy', allow_pickle=True)
    
    #Find maxima within the weightmaps, output their coordinates and scores
    line = np.linspace(0,size,100)

    x1,y1,z1 = score_map(map1, threshold=0.5, N = 3)
    xmax1, ymax1 = line[x1], -line[y1]

    x2,y2,z2 = score_map(map2, threshold=0.5, N = 3)
    xmax2, ymax2 = line[x2], -line[y2]

    x3,y3,z3 = score_map(map3, threshold=0.5, N = 3)
    xmax3, ymax3 = line[x3], -line[y3]

    #Get the einstein radius
    eR1 = find_eR(weights1, x1, y1, np.linspace(1,60,size))
    eR2 = find_eR(weights2, x2, y2, np.linspace(1,60,size))
    eR3 = find_eR(weights3, x3, y3, np.linspace(1,60,size))

    #Plot the results
    fig, ax = plt.subplots(1,3, figsize=(10,5), sharex=True, sharey=True)
    fig.suptitle('A2744', fontsize=16)

    plots.plot_likelihood_map(ax[0],np.log10(map1+10**-10),None,None,xmax1,ymax1,100*z1,eR1,size,'Shear')
    plots.plot_likelihood_map(ax[1],np.log10(map2+10**-10),None,None,xmax2,ymax2,100*z2,eR2,size,'Flexion')
    plots.plot_likelihood_map(ax[2],np.log10(map3+10**-10),None,None,xmax3,ymax3,100*z3,eR3,size,'Flexion + Shear')

    plt.savefig('Images/a2744.png')


def create_varied_tests():
    size = 50
    Nlens = [1,2]
    #Create lens
    xl = [20,-20]
    yl = [0,0]
    eR = [5,5]

    #Vary this by number of sources
    Nsources = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,200]
    xs, ys = make_sources(Nsources[-1], size = size)

    noise = [True, False]

    for Nsource in Nsources:
        source_x = xs[:Nsource]
        source_y = ys[:Nsource]
        for Nlen in Nlens:
            lenses = Lens(xl[:Nlen], yl[:Nlen], eR[:Nlen])
            for n in noise:
                if n:
                    sources = Source(source_x, source_y, np.random.normal(0,sigma_g,Nsource), np.random.normal(0,sigma_g,Nsource), np.random.normal(0,sigma_f,Nsource), np.random.normal(0,sigma_f,Nsource))
                else:
                    sources = Source(source_x, source_y, np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource))

                sources.calc_shear(lenses)
                sources.calc_flex(lenses)

                weights1, weights2, weights3 = sources.weights(size,sigma_f=10**-2,sigma_g=10**-3)

                map1 = process_weights(weights1, np.linspace(1,60,size), size)
                map2 = process_weights(weights2, np.linspace(1,60,size), size)
                map3 = process_weights(weights3, np.linspace(1,60,size), size)

                #Find maxima within the weightmaps, output their coordinates and scores
                line = np.linspace(-size,size,size)

                x1,y1,z1 = score_map(map1, threshold=0.5)
                xmax1, ymax1 = line[x1], -line[y1]

                x2,y2,z2 = score_map(map2, threshold=0.5)
                xmax2, ymax2 = line[x2], -line[y2]

                x3,y3,z3 = score_map(map3, threshold=0.5)
                xmax3, ymax3 = line[x3], -line[y3]

                #Get the einstein radius
                eR1 = find_eR(weights1, x1, y1, np.linspace(1,60,size))
                eR2 = find_eR(weights2, x2, y2, np.linspace(1,60,size))
                eR3 = find_eR(weights3, x3, y3, np.linspace(1,60,size))

                #Plot the results
                fig, ax = plt.subplots(1,3, figsize=(15,10), sharex=True, sharey=True)
                fig.suptitle('Likelihood Maps \n Nlens = {}, Nsource = {} \n Noise = {}'.format(Nlen, Nsource, n), fontsize=16)

                plots.plot_likelihood_map(ax[0],np.log10(map1+10**-20),lenses,sources,xmax1,ymax1,100*z1,eR1, size,'Shear')
                plots.plot_likelihood_map(ax[1],np.log10(map2+10**-20),lenses,sources,xmax2,ymax2,100*z2,eR2, size,'Flexion')
                plots.plot_likelihood_map(ax[2],np.log10(map3+10**-20),lenses,sources,xmax3,ymax3,100*z3,eR3, size,'Flexion + Shear')

                #Shift the axes up a bit to remove whitespace
                plt.subplots_adjust(top=1)

                plt.savefig('Images/tests/{}_map_lens_{}_source_{}.png'.format('noiseless' if n == False else 'noisy', Nlen, Nsource))
                plt.close()

        print('Done with {} sources'.format(Nsource))


if __name__ == "__main__":
    #Create a very simple lensing field

    size = 50
    res = 100
    Nlens = 1
    Nsource = 4

    #Create lens
    xl = [0]
    yl = [0]
    eR = [5]

    #Create sources
    xs = np.array([20, 20, -20, -20])
    ys = np.array([20, -20, 20, -20])

    noise1 = 10**-4
    noise2 = 10**-4
    sources = Source(xs, ys, np.random.normal(0,noise1,Nsource), np.random.normal(0,noise1,Nsource), np.random.normal(0,noise2,Nsource), np.random.normal(0,noise2,Nsource))
    lenses = Lens(xl, yl, eR)

    sources.calc_shear(lenses)
    sources.calc_flex(lenses)

    weights1, weights2, weights3 = sources.weights(size)

    map1 = process_weights(weights1, np.linspace(1,60,res), res)
    map2 = process_weights(weights2, np.linspace(1,60,res), res)
    map3 = process_weights(weights3, np.linspace(1,60,res), res)

    #Find maxima within the weightmaps, output their coordinates and scores
    line = np.linspace(-size,size,res)

    x1,y1,z1 = score_map(map1, threshold=0.1)
    xmax1, ymax1 = line[x1], -line[y1]

    x2,y2,z2 = score_map(map2, threshold=0.1)
    xmax2, ymax2 = line[x2], -line[y2]

    x3,y3,z3 = score_map(map3, threshold=0.1)
    xmax3, ymax3 = line[x3], -line[y3]

    #Get the einstein radius
    eR1 = find_eR(weights1, x1, y1, np.linspace(1,60,res))
    eR2 = find_eR(weights2, x2, y2, np.linspace(1,60,res))
    eR3 = find_eR(weights3, x3, y3, np.linspace(1,60,res))

    #We can ask the question, given the flexion of each source, what einstein radius would that source
    #predict at the maximum?
    #We can then plot this as a function of the einstein radius
    fig, ax = plt.subplots(1,3, figsize=(10,5), sharex=True, sharey=True)

    for i in range(Nsource):
        F = np.sqrt(sources.f1[i]**2 + sources.f2[i]**2)
        dist1 = np.sqrt((xs[i] - xmax1)**2 + (ys[i] - ymax1)**2)
        dist2 = np.sqrt((xs[i] - xmax2)**2 + (ys[i] - ymax2)**2)
        dist3 = np.sqrt((xs[i] - xmax3)**2 + (ys[i] - ymax3)**2)

        predict1 = 2*dist1*np.abs(dist1) * F
        predict2 = 2*dist2*np.abs(dist2) * F       
        predict3 = 2*dist3*np.abs(dist3) * F

        ax[0].scatter(eR1, predict1, label='Source {}'.format(i))
        ax[1].scatter(eR2, predict2, label='Source {}'.format(i))
        ax[2].scatter(eR3, predict3, label='Source {}'.format(i))

    name = ['Shear', 'Flexion', 'Flexion + Shear']
    for i in range(3):
        ax[i].plot(np.linspace(0,20,100), np.linspace(0,20,100), label='1:1')
        ax[i].set_xlabel(r'Located $\theta_E$')
        ax[i].set_ylabel(r'Predicted $\theta_E$')
        ax[i].legend()
        ax[i].set_title(name[i])
        ax[i].set_xlim([0,20])
        ax[i].set_ylim([0,20])
        ax[i].set_aspect('equal', adjustable='box')


    #Plot the results
    fig, ax = plt.subplots(1,3, figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Likelihood Maps \n Nlens = {}, Nsource = {} \n Noise = {}'.format(Nlens, Nsource, True), fontsize=16)

    plots.plot_likelihood_map(ax[0],np.log10(map1+10**-10),lenses,sources,xmax1,ymax1,100*z1,eR1, size,'Shear')
    plots.plot_likelihood_map(ax[1],np.log10(map2+10**-10),lenses,sources,xmax2,ymax2,100*z2,eR2, size,'Flexion')
    plots.plot_likelihood_map(ax[2],np.log10(map3+10**-10),lenses,sources,xmax3,ymax3,100*z3,eR3, size,'Flexion + Shear')

    #Shift the axes up a bit to remove whitespace
    plt.subplots_adjust(top=1)

    plt.show()
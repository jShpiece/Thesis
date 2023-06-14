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
sigma_f = 10**-2
sigma_g = 10**-3


def make_sources(Nsource, size = 100):
    #Create a set of sources
    #Uniformly distributed in a circle of radius size/2 - size gives the length of the side of the square
    rs = np.sqrt(np.random.random(Nsource)) * (size / 2)
    thetas = np.random.random(Nsource) * 2*np.pi
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    return xs, ys


def noise_variance():
    #Generates the likelihood maps for a single source with different
    #estimated noise levels
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


def make_lensing_field(Nlens, size = 50, Noise = True, centered = False):
    Nsource = int(10**-2 * size**2)
    xs, ys = make_sources(Nsource, size = size)

    if centered:
        xl = [0]
        yl = [0]
        eR = [5]
    else:
        xl = np.random.uniform(-size,size,Nlens)
        yl = np.random.uniform(-size,size,Nlens)
        eR = np.random.uniform(0,10,Nlens)

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


def create_test_set(Nlens, Noise = True, centered = True, size = 50):
    #Generate a test set of lenses and sources
    #Return the weights for each weighting scheme and the coordinates of the maxima

    #Note: here 1,2,3 correspond to shear, flexion, and shear + flexion respectively
    lenses,sources = make_lensing_field(Nlens, size = size, Noise = Noise, centered = centered)

    weights1, weights2, weights3 = sources.weights(size,sigma_f=10**-2,sigma_g=10**-3)
    weights = [weights1, weights2, weights3]

    np.save('Data/test_weights_Nlens_{}.npy'.format(Nlens), weights)

    map1 = process_weights(weights1, np.linspace(1,60,size), size)
    map2 = process_weights(weights2, np.linspace(1,60,size), size)
    map3 = process_weights(weights3, np.linspace(1,60,size), size)

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

    eR1 = find_eR(weights[0], x1, y1, eR_range = np.linspace(1,60,size))
    eR2 = find_eR(weights[1], x2, y2, eR_range = np.linspace(1,60,size))
    eR3 = find_eR(weights[2], x3, y3, eR_range = np.linspace(1,60,size))

    eR = [eR1, eR2, eR3]

    return maps, lenses, sources, maxima, strengths, eR


def plot_test_map(Noise=False, centered = True, size = 50, Nlens = 1):
    maps, lenses, sources, maxima, strengths, eR = create_test_set(Nlens, Noise = Noise, centered = centered, size = size)
    #Plot the results
    fig, ax = plt.subplots(1,3, figsize=(10,7), sharex=True, sharey=True)
    fig.suptitle('{} Map'.format('Noiseless' if Noise == False else 'Noisy'), y = 0.9, fontsize=16)
    
    plots.plot_likelihood_map(ax[0],np.log10(maps[0]+10**-20),lenses,None,maxima[0][0],maxima[0][1],100*strengths[0],eR[0],size,'Shear')
    plots.plot_likelihood_map(ax[1],np.log10(maps[1]+10**-20),lenses,None,maxima[1][0],maxima[1][1],100*strengths[1],eR[1],size,'Flexion')
    plots.plot_likelihood_map(ax[2],np.log10(maps[2]+10**-20),lenses,None,maxima[2][0],maxima[2][1],100*strengths[2],eR[2],size,'Flexion + Shear')

    #Shift the axes up a bit to remove whitespace
    plt.subplots_adjust(top=1)
    plt.savefig('Images/test_map_{}_Nlens_{}.png'.format('Noiseless' if Noise == False else 'Noisy', Nlens))


def random_realization(size = 100):
    #Create a random realization of a map
    #This is a function that can be run in parallel
    #It returns the coordinates of the maxima closest to the lens
    #For each weighting scheme
    #It also returns the coordinates of the lens
    #And the coordinates of the maxima
    #And the scores of the maxima
    #This is so we can plot the results later

    
    *_, xmax1, xmax2, xmax3, ymax1, ymax2, ymax3, z1, z2, z3 = create_test_set(1, Noise = True, centered = True, size = size, res = size)

    #Return the maxima with the highest score
    max1 = np.argmax(z1)
    max2 = np.argmax(z2)
    max3 = np.argmax(z3)

    #Return the coordinates of the maxima
    x1 = xmax1[max1]
    y1 = ymax1[max1]
    x2 = xmax2[max2]
    y2 = ymax2[max2]
    x3 = xmax3[max3]
    y3 = ymax3[max3]   

    return x1, y1, x2, y2, x3, y3


def run_random_realization(Ntrials, size = 50):
    #Run the random realization function
    start = time.time()

    pbar = tqdm(total=Ntrials)

    pool = Pool()
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
    x2 = np.array([result[2] for result in results])
    y2 = np.array([result[3] for result in results])
    x3 = np.array([result[4] for result in results])
    y3 = np.array([result[5] for result in results])

    #Plot the results
    fig, ax = plt.subplots(1,3, figsize=(10,5))
    fig.suptitle('Noiseless Random Realization: Nsource = {}'.format(len(x1)), fontsize=16)

    plots.plot_param_conf(x1,y1,ax[0],['Data: x = {:.2f} $\pm$ {:.2f}, y = {:.2f} $\pm$ {:.2f}'.format(
                        np.mean(x1),np.std(x1)/np.sqrt(len(x1)),np.mean(y1),np.std(y1)/np.sqrt(len(x1))), 'x', 'y'],'Shear')

    plots.plot_param_conf(x2,y2,ax[1],['Data: x = {:.2f} $\pm$ {:.2f}, y = {:.2f} $\pm$ {:.2f}'.format(
                        np.mean(x2),np.std(x2)/np.sqrt(len(x2)),np.mean(y2),np.std(y2)/np.sqrt(len(x2))), 'x', 'y'],'Flexion')
    
    plots.plot_param_conf(x3,y3,ax[2],['Data: x = {:.2f} $\pm$ {:.2f}, y = {:.2f} $\pm$ {:.2f}'.format(
                        np.mean(x3),np.std(x3)/np.sqrt(len(x3)),np.mean(y3),np.std(y3)/np.sqrt(len(x3))), 'x', 'y'],'Flexion + Shear')

    end = time.time()
    print('Time: {:.2f} s'.format(end-start))
    plt.savefig('Images/random_realization_N_{}.png'.format(Ntrials))


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
    size = 1.1 * np.max(np.sqrt((x[:,np.newaxis] - x)**2 + (y[:,np.newaxis] - y)**2))

    #Create the sources
    sources = Source(x, y, g1, g2, f1, f2)

    #Compute the weights
    weights1, weights2, weights3 = sources.weights(size,100,sigma_f=10**-2,sigma_g=10**-3)
    
    np.save('Data/a2744_weights.npy', [weights1, weights2, weights3])
    #weights1, weights2, weights3 = np.load('a2744_weights.npy', allow_pickle=True)
    
    #Find maxima within the weightmaps, output their coordinates and scores
    line = np.linspace(0,size,100)

    x1,y1,z1 = score_map(weights1, threshold=0.5, N = 3)
    xmax1, ymax1 = line[x1], -line[y1]

    x2,y2,z2 = score_map(weights2, threshold=0.5, N = 3)
    xmax2, ymax2 = line[x2], -line[y2]

    x3,y3,z3 = score_map(weights3, threshold=0.5, N = 3)
    xmax3, ymax3 = line[x3], -line[y3]

    #Plot the results
    fig, ax = plt.subplots(1,3, figsize=(10,5), sharex=True, sharey=True)
    fig.suptitle('A2744', fontsize=16)

    plots.plot_likelihood_map(ax[0],np.log10(weights1+10**-10),None,None,xmax1,ymax1,100*z1,size,'Shear')
    plots.plot_likelihood_map(ax[1],np.log10(weights2+10**-10),None,None,xmax2,ymax2,100*z2,size,'Flexion')
    plots.plot_likelihood_map(ax[2],np.log10(weights3+10**-10),None,None,xmax3,ymax3,100*z3,size,'Flexion + Shear')

    plt.savefig('Images/a2744.png')


if __name__ == "__main__":
    #noise_variance()
    start = time.time()
    plot_test_map(Noise = False, centered = True, size = 20, Nlens = 1)
    end = time.time()
    print('Noiseless Lens Time: {:.2f} s'.format(end-start))
    start = time.time()
    plot_test_map(Noise = True, centered = True, size = 20, Nlens = 1)
    end = time.time()
    print('Noisy Lens Time: {:.2f} s'.format(end-start))
    #run_random_realization(100, size = 100)
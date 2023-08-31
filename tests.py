#This file holds code that test our pipeline - pipeline.py
#It tests the functions in pipeline.py
#This allows me to reconstruct any test images
#Or run any tests I want to run

from pipeline import Source, Lens, score_map
import utils
import plots
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.signal import convolve


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


def flexion_noise_test(Ntrials):
    #Look at how the choice of input noise and estimated noise affects the results
    size = 50
    res = size*2
    Nsource = 4

    #Create sources as float64 arrays
    #xs, ys = make_sources(Nsource, size = size)
    xs = np.array([20, 20, -20, -20])
    ys = np.array([20, -20, 20, -20])
    #Change the data type to float64 if it isn't already
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    #Create lens
    xl = [0]
    yl = [0]
    eR = [5]

    lenses = Lens(xl, yl, eR)

    #Input noise
    noise_input = np.logspace(-5,-1,Ntrials)
    #Estimated noise
    noise_est = np.logspace(-5,-1,Ntrials)

    output = np.zeros((Ntrials,Ntrials))
    for i in range(Ntrials):
        for j in range(Ntrials):
            sigma = np.exp(np.log(noise_input[i]))
            sources = Source(xs,ys,np.random.normal(0,sigma,Nsource), np.random.normal(0,sigma,Nsource), np.random.normal(0,sigma,Nsource), np.random.normal(0,sigma,Nsource))
            sources.calc_flex(lenses)

            noise = np.exp(np.log(noise_est[j]))
            weights = sources.weights(size,sigma_f=noise,sigma_g=noise, eRmin=1, eRmax=60)
            weights = [weights[1]] #Only look at flexion
            maps = utils.process_weights(weights, np.linspace(1,60,res))

            #Locate maxima and get the einstein radius
            max_coords = score_map(maps, threshold=0.5)
            eR_coords = utils.find_eR(weights, max_coords)

            max_index = np.argmax(max_coords[0][2])
            output[i,j] = np.round(eR_coords[0][max_index],2)
    
    #Plot the results
    fig, ax = plt.subplots(1,1, figsize=(10,10), sharex=True, sharey=True)
    fig.suptitle('Flexion Noise Test', fontsize=16)

    im = ax.imshow(output, cmap='viridis', origin='lower', extent=[-1, -5, -1, -5])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Input Noise (log scale)')
    ax.set_ylabel('Estimated Noise (log scale)')
    ax.set_title('Einstein Radius')
    plt.show()


def minimization_test():
    #This function holds the code I wrote to test minimization approaches
    start = time.time()
    size = 50
    res = size*2
    Nsource = 4
    Nlens = 1

    line = np.linspace(-size,size,res)
    eR_range = np.linspace(1,30,res)

    xs = np.array([20, 20, -20, -20]) 
    ys = np.array([20, -20, 20, -20]) 
    #Change the data type to float64 if it isn't already
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    #Create lens
    xl = [0]
    yl = [0]
    eR = [5]

    lenses = Lens(xl, yl, eR)

    shear_noise = 10**-3
    flex_noise = 10**-3
    sources = Source(xs,ys,np.random.normal(0,shear_noise,Nsource), np.random.normal(0,shear_noise,Nsource), 
                     np.random.normal(0,flex_noise,Nsource), np.random.normal(0,flex_noise,Nsource))
    sources.calc_flex(lenses)
    sources.calc_shear(lenses)
    shear_weights, flex_weights = sources.weights(size,sigma_f=flex_noise,sigma_g=shear_noise, eRmin=1, eRmax=30)
    
    true_weights = []
    true_weights.append(np.exp(np.sum(np.log(np.array(shear_weights)), axis=0)))
    true_weights.append(np.exp(np.sum(np.log(np.array(flex_weights)), axis=0)))
    
    #Okay, now try to reconstruct the weights by varying lens parameters, computing the weights, and then comparing the results
    #to the true weights. The goal is to find the lens parameters that minimize the difference between the true weights and the
    #reconstructed weights

    #First, we need to define a function that takes in a set of lens parameters and returns the weights
    def get_weights(params):
        #params is a list of lens parameters
        test_lens = Lens(params[0], params[1], params[2])
        test_sources = Source(xs,ys,np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource))
        test_sources.calc_flex(test_lens)
        test_sources.calc_shear(test_lens)
        test_weights = test_sources.weights(size,sigma_f=flex_noise,sigma_g=shear_noise, eRmin=1, eRmax=30)
        return test_weights
    
    #Now perform a chi-squared minimization to find the best fit parameters
    #We need to define a function that takes in a set of lens parameters and returns the chi-squared value
    def chi_squared(params, sigma1, sigma2):
        shear_test, flex_test = get_weights(params) #Get the weights
        shear_test = np.exp(np.sum(np.log(np.array(shear_test)), axis=0)) #Take the product of the weights in log space
        flex_test = np.exp(np.sum(np.log(np.array(flex_test)), axis=0))

        chi2_shear = np.sum(((shear_test - true_weights[0])**2) / true_weights[0]) #Compute the chi-squared value
        chi2_flex = np.sum(((flex_test - true_weights[1])**2) / true_weights[1])
        print(chi2_shear, chi2_flex)
        return chi2_shear, chi2_flex
    
    #We're going to try a brute - force approach
    #We'll try every combination of lens parameters in a grid
    #We'll then evaluate the chi-squared value for each combination
    #And then find the minimum

    xl_range = np.linspace(-size, size, 5) + 0.5
    yl_range = np.linspace(-size, size, 5) + 0.5
    eR_range = np.linspace(1, 30, 20)

    chi2_shear = np.zeros((len(xl_range), len(yl_range)))
    chi2_flex = np.zeros((len(xl_range), len(yl_range)))

    for i in range(len(xl_range)):
        for j in range(len(yl_range)):
            params = [[xl_range[i]], [yl_range[j]], [5]]
            chi21, chi22 = chi_squared(params, shear_noise, flex_noise)
            chi2_shear[i,j] = chi21
            chi2_flex[i,j] = chi22
    
    #Locate minima
    min_shear = np.unravel_index(np.argmin(chi2_shear), chi2_shear.shape)
    min_flex = np.unravel_index(np.argmin(chi2_flex), chi2_flex.shape)

    xmin_shear = xl_range[min_shear[0]]
    ymin_shear = yl_range[min_shear[1]]

    xmin_flex = xl_range[min_flex[0]]
    ymin_flex = yl_range[min_flex[1]]
    
    #Now plot
    fig, ax = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    fig.suptitle('Chi-Squared Maps: Fixed eR', fontsize=16)

    im1 = ax[0].imshow(np.log(chi2_shear), extent=[-size,size,-size,size])
    fig.colorbar(im1, ax=ax[0])
    ax[0].set_aspect('equal', 'box') #This is a map, so we want the aspect ratio to be equal
    ax[0].set_title('Shear')
    ax[0].scatter(xs,ys, marker='x', color='black', label='Source') #Mark the source positions 
    ax[0].scatter(xl,yl, marker='*', color='black', label='Lens') #Mark the lens position
    ax[0].scatter(xmin_shear, ymin_shear, marker='o', color='red', label='Minimum') #Mark the minimum
    ax[0].set_xlabel('xl')
    ax[0].set_ylabel('yl')
    ax[0].legend()

    im2 = ax[1].imshow(np.log(chi2_flex), extent=[-size,size,-size,size])
    fig.colorbar(im2, ax=ax[1])
    ax[1].set_aspect('equal', 'box') #This is a map, so we want the aspect ratio to be equal
    ax[1].set_title('Flexion')
    ax[1].scatter(xs,ys, marker='x', color='black', label='Source') #Mark the source positions
    ax[1].scatter(xl,yl, marker='*', color='black', label='Lens') #Mark the lens position
    ax[1].scatter(xmin_flex, ymin_flex, marker='o', color='red', label='Minimum') #Mark the minimum
    ax[1].set_xlabel('xl')
    ax[1].set_ylabel('yl')
    ax[1].legend()

    plt.savefig('Images/chi_squared_map.png')
    
    eR_chi2_shear = np.zeros(len(eR_range))
    eR_chi2_flex = np.zeros(len(eR_range))
    
    for i in range(len(eR_range)):
        params = [[0], [0], [eR_range[i]]]
        chi21, chi22 = chi_squared(params, shear_noise, flex_noise)
        eR_chi2_shear[i] = chi21
        eR_chi2_flex[i] = chi22
        
    #Locate minima
    shear_min = np.argmin(eR_chi2_shear)
    flex_min = np.argmin(eR_chi2_flex)
    
    #Now plot
    plt.figure()
    plt.plot(eR_range, eR_chi2_shear, label='Shear')
    plt.scatter(eR_range[shear_min], eR_chi2_shear[shear_min], marker='o', color='red', label='Min Shear')
    plt.plot(eR_range, eR_chi2_flex, label='Flexion')
    plt.scatter(eR_range[flex_min], eR_chi2_flex[flex_min], marker='o', color='red', label='Min Flexion')
    plt.legend()
    plt.xlabel('eR')
    plt.ylabel(r'$\chi^2$')
    plt.yscale('log')
    
    plt.savefig('Images/chi_squared_eR.png')
    
    plt.show()




    
    '''
    #Now perform the minimization
    #We need to define a starting point
    xl0 = 0
    yl0 = 0
    eR0 = 5
    #guess = [xl0, yl0, eR0] #This is the true lens position
    #guess = [xl0 + 1, yl0 + 1, eR0 + 1] #This is a guess that is close to the true lens position
    guess = [xl0, yl0, eR0 + 10] #This is a guess that is far from the true lens position

    #Limit minimization to a maximum number of iterations
    N_iter = 10
    #Set bounds on the parameters
    xl_bounds = (-size, size)
    yl_bounds = (-size, size)
    eR_bounds = (1, 60)
    bounds = [xl_bounds, yl_bounds, eR_bounds]
    #Relax the tolerance
    tol = 1e-3
    #Set the method - be sure to choose one that allows bounds
    method = 'Powell'

    #Now perform the minimization
    shear_result = minimize(chi_squared, guess, args=(0, shear_noise), method=method, bounds=bounds, tol=tol, options={'maxiter':N_iter})
    flex_result = minimize(chi_squared, guess, args=(1, flex_noise), method=method, bounds=bounds, tol=tol,  options={'maxiter':N_iter})
    print('Minimization complete')

    print('Shear result')
    print(shear_result)
    print('Flexion result')
    print(flex_result)
    stop = time.time()
    print('Time taken: {} seconds'.format(stop-start))
    '''



if __name__ == '__main__':
    minimization_test()
    raise SystemExit

    start = time.time()
    size = 50
    res = size*2
    Nsource = 4
    Nlens = 1

    line = np.linspace(-size,size,res)
    eR_range = np.linspace(1,60,res)

    xs = np.array([20, 20, -20, -20]) / 2
    ys = np.array([20, -20, 20, -20]) / 2
    #xs = np.array([25])
    #ys = np.array([20])
    #Change the data type to float64 if it isn't already
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    #xs,ys = make_sources(20, size = size)

    #Correct Nsource if necessary
    Nsource = len(xs) #this just avoids throwing errors if I set source positions manually

    #Create lens
    xl = [0]
    yl = [0]
    eR = [5]

    lenses = Lens(xl, yl, eR)

    shear_noise = 10**-3
    flex_noise = 10**-3

    estimated_noise = shear_noise * np.array([1, 2, 5, 10, 20, 50, 100])
    
    sources = Source(xs,ys,np.random.normal(0,shear_noise,Nsource), np.random.normal(0,shear_noise,Nsource), 
                        np.random.normal(0,flex_noise,Nsource), np.random.normal(0,flex_noise,Nsource))
    sources.calc_flex(lenses)
    sources.calc_shear(lenses)
    

    for n in range(len(estimated_noise)):
        shear_weights, flex_weights = sources.weights(size,sigma_f=estimated_noise[n],sigma_g=estimated_noise[n], eRmin=1, eRmax=60)

        #Shear
        product_shear_weights = np.exp(np.sum(np.log(np.array(shear_weights)), axis=0)) #This is the product of the weights, taken as the sum in log space
        product_shear_maps = utils.process_weights([product_shear_weights], np.linspace(1,60,res))[0]
        #Lets find the maxima
        shear_maxima = score_map([product_shear_maps], threshold=0.5)
        shear_eR = utils.find_eR([product_shear_weights], shear_maxima)[0]

        #Flexion
        product_flex_weights = np.exp(np.sum(np.log(np.array(flex_weights)), axis=0))
        product_flex_maps = utils.process_weights([product_flex_weights], np.linspace(1,60,res))[0]
        #Lets find the maxima
        flex_maxima = score_map([product_flex_maps], threshold=0.5)
        flex_eR = utils.find_eR([product_flex_weights], flex_maxima)[0]

        
        #Now plot the results
        fig, ax = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
        fig.suptitle('Likelihood Distributions: Different Merging Methods \n noise / estimated noise = {:.2E}'
                     .format(flex_noise/estimated_noise[n]), fontsize=16)


        ax[0].imshow(np.log10(product_shear_maps+1e-20), extent=[-size,size,-size,size])
        ax[0].set_aspect('equal', 'box') #This is a map, so we want the aspect ratio to be equal
        ax[0].set_title('Shear')
        ax[0].scatter(xs,ys, marker='x', color='black', label='Source') #Mark the source positions
        ax[0].scatter(xl,yl, marker='*', color='black', label='Lens') #Mark the lens position
        for i in range(len(shear_eR)):
            #Mark the maxima, with the associated einstein radius
            ax[0].scatter(line[shear_maxima[0][0][i]], line[shear_maxima[0][1][i]], 
                          marker='o', color='red', label='Maxima {}: {:.2f}'.format(i+1, eR_range[shear_eR[i]]))
        ax[0].legend(loc = 'upper right')
        
        ax[1].imshow(np.log10(product_flex_maps+1e-20), extent=[-size,size,-size,size])
        ax[1].set_aspect('equal', 'box')
        ax[1].set_title('Flexion')
        ax[1].scatter(xs,ys, marker='x', color='black', label='Source')
        ax[1].scatter(xl,yl, marker='*', color='black', label='Lens')
        for i in range(len(flex_eR)):
            ax[1].scatter(line[flex_maxima[0][0][i]], line[flex_maxima[0][1][i]], 
                          marker='o', color='red', label='Maxima {}: {:.2f}'.format(i+1, eR_range[flex_eR[i]]))
        ax[1].legend(loc = 'upper right')

        plt.savefig('Images/weight_merging_{}sources_noise_{}.png'.format(Nsource,n))
        plt.show()
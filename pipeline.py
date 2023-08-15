import numpy as np
from scipy import integrate
from numba import njit, prange

class Source:
    '''
    This class represents a source galaxy. It contains the following attributes:
    x: x-coordinates of the source galaxy
    y: y-coordinates of the source galaxy
    gamma1: gamma1 component of the shear
    gamma2: gamma2 component of the shear
    f1: f1 component of the flexion
    f2: f2 component of the flexion

    The class also contains the following methods:
    calc_shear: calculates the shear at the source galaxy due to the lenses
    calc_flex: calculates the flexion at the source galaxy due to the lenses
    weights: calculates the likelihood of each lensing configuration in a 3D parameter space
    '''

    def __init__(self, x, y, gamma1, gamma2, f1, f2):
        self.x = x
        self.y = y
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.f1 = f1
        self.f2 = f2
    
    
    def calc_shear(self, lenses):
        Nl = len(lenses.x)
        for n in range(Nl):
            dx = self.x - lenses.x[n]
            dy = self.y - lenses.y[n]
            r = np.sqrt(dx ** 2 + dy ** 2 + 0.5 ** 2)
            phi = np.arctan2(dy, dx)
            self.gamma1 -= lenses.eR[n] * np.cos(2 * phi) / (2 * r)
            self.gamma2 -= lenses.eR[n] * np.sin(2 * phi) / (2 * r)


    def calc_flex(self, lenses):
        Nl = len(lenses.x)
        for n in range(Nl):
            dx = self.x - lenses.x[n]
            dy = self.y - lenses.y[n]
            r2 = dx ** 2 + dy ** 2 + 0.5 ** 2
            phi = np.arctan2(-dy, dx)
            self.f1 -= lenses.eR[n] * np.cos(phi) / (2 * r2)
            self.f2 -= lenses.eR[n] * np.sin(phi) / (2 * r2)


    def weights(self, size, sigma_f = 10**-2, sigma_g = 10**-3, eRmin = 1, eRmax = 60):
        # This function calculates the likelihood of each lensing configuration in a 3D parameter space
        # size: 1/2 size of the grid in arcseconds
        # sigma_f: standard deviation of the flexion
        # sigma_g: standard deviation of the shear
        # eRmin: minimum allowed value of eR
        # eRmax: maximum allowed value of eR

        #Initialize the grid
        xs, ys, F1, F2, gamma1, gamma2 = self.x, self.y, self.f1, self.f2, self.gamma1, self.gamma2 # Read in the source properties
        res = size * 2
        line = np.linspace(-size,size,res)
        eR_range = np.linspace(eRmin + 0.2, eRmax,res)
        x,y = np.meshgrid(line,line)

        #Compute the shear and flexion for each source (magnitude and angle)
        F = np.sqrt(F1**2 + F2**2)
        phiF = np.arctan2(F2,F1) + np.pi
        gamma = np.sqrt(gamma1**2 + gamma2**2)
        phi_gamma = np.arctan2(gamma2,gamma1) / 2 + np.pi

        #Initialize the weights
        weights1 = np.zeros((res,res,res))
        weights2 = np.zeros((res,res,res))

        for n in range(len(xs)):
            #Adjust the coordinates to center the source
            xn = x - xs[n]
            yn = y + ys[n] # The y-axis is flipped
            r = np.sqrt(xn**2 + yn**2)
            phi1 = np.arctan2(-yn,xn) - phi_gamma[n]
            phi2 = np.arctan2(-yn,xn) + phiF[n] + np.pi/2

            shear_contribution = compute_weights(gamma[n], 'shear', r, phi1, eR_range, res, sigma_g, eRmin, eRmax)
            flexion_contribution = compute_weights(F[n], 'flexion', r, phi2, eR_range, res, sigma_f, eRmin, eRmax)

            if np.sum(shear_contribution) > 0:
                weights1 += np.log(shear_contribution)

            if np.sum(flexion_contribution) > 0:
                weights2 += np.log(flexion_contribution)
            
        #Return to linear space
        weights1 = np.exp(weights1)
        weights2 = np.exp(weights2)
        #Normalize the weights
        weights1 /= np.sum(weights1)
        weights2 /= np.sum(weights2)

        weights3 = weights1 * weights2
        weights3 /= np.sum(weights3)
        
        return [weights1, weights2, weights3]


class Lens:
    '''
    This class represents a lens object. It contains the following attributes:
    x: x-coordinates of the lens
    y: y-coordinates of the lens
    eR: einstein radius of the lens
    '''
    def __init__(self, x, y, eR):
        self.x = x
        self.y = y
        self.eR = eR


def flexion_integrand(eR, F, r, sigma, phi):
    lens_F = -(eR * np.cos(phi)) / (2 * r**2)
    gaussian_term = np.exp((-(F - lens_F)**2) / (2 * sigma**2))
    power_term = np.abs(eR)**-0.95
    return gaussian_term * power_term


def shear_integrand(eR, gamma, r, sigma, phi):
    lens_gamma = -(eR * np.cos(2 * phi)) / (2 * r)
    gaussian_term = np.exp((-(gamma - lens_gamma)**2) / (2 * sigma**2))
    power_term = np.abs(eR)**-0.95
    return gaussian_term * power_term


def compute_weights(signal, signal_type, r, phi, eR, res, sigma, eRmin=1, eRmax=60):
    denominator = np.zeros((res, res))
    weights = np.zeros((res, res, res))

    if signal_type == 'flexion':
        integrand = flexion_integrand
        filter = np.exp(-r / 20) # Flexion will not be considered beyond 20 arcseconds
        coefficient = 2 * filter * r / np.abs(np.cos(phi))
    elif signal_type == 'shear':
        integrand = shear_integrand
        coefficient = 2 * r / np.abs(np.cos(2 * phi))

    # Cache integrand function call
    integrand_vals = integrand(eR[:, None, None], signal, r, sigma, phi)
    
    # Precompute coefficient and small constant
    small_const = 1e-5
    
    # Cache denominator function call
    for i in range(res):
        for j in range(res):
            denominator[i, j] = integrate.quad(integrand, eRmin, eRmax, args=(signal, r[i, j], sigma, phi[i, j]))[0] 
            denominator += small_const
 
    # Adjust coefficient to have the correct shape 
    # Coefficient = coefficient[i,j], weights = weights[k,i,j]
    coefficient = coefficient[:, :, None]
    # Calculate weights using cached values
    weights = coefficient * integrand_vals / denominator + small_const

    return weights


def score_map(maps, threshold=0.1):
    '''
    This function takes in a colormap and returns the locations of the peaks in the colormap
    as well as the scores associated with each peak. It also takes in a threshold value, which
    is used to determine the minimum score threshold. The minimum score threshold is defined as
    the maximum score in the colormap multiplied by the threshold value. The function returns
    the locations of the peaks and the scores associated with each peak that are above the minimum
    score threshold.
    '''
    coords = []
    for i in range(len(maps)):
        colormap = maps[i]
        # Step 1: Identify local maxima
        local_maxima = np.zeros_like(colormap, dtype=bool)
        local_maxima[(colormap >= np.roll(colormap, 1, axis=0)) &
                    (colormap >= np.roll(colormap, -1, axis=0)) &
                    (colormap >= np.roll(colormap, 1, axis=1)) &
                    (colormap >= np.roll(colormap, -1, axis=1))] = True

        # Step 2: Determine the width of regions
        widths = np.zeros_like(colormap)
        for i, j in np.transpose(np.where(local_maxima)):
            peak_value = colormap[i, j]
            contour = np.where(colormap >= threshold * peak_value)
            distances = np.sqrt((contour[0] - i)**2 + (contour[1] - j)**2)
            widths[i, j] = np.max(distances)

        # Step 3: Rank the locations
        scores = colormap * widths

        # Set the minimum score threshold
        score_threshold = np.max(scores) * 0.5

        relevant_locations = np.transpose(np.where(scores >= score_threshold))

        #Unpack the relevant locations
        yloc = relevant_locations[:,0]
        xloc = relevant_locations[:,1]

        #Get the scores associated with each location
        scores = scores[yloc,xloc]
    
        coords.append([xloc, yloc, scores])

    return coords
import numpy as np
from scipy import integrate

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
        self.x = x[:]
        self.y = y[:]
        self.gamma1 = gamma1[:]
        self.gamma2 = gamma2[:]
        self.f1 = f1[:]
        self.f2 = f2[:]
    

    def calc_shear(self, lenses):
        dx = self.x[:, np.newaxis] - lenses.x
        dy = self.y[:, np.newaxis] - lenses.y
        r = np.sqrt(dx ** 2 + dy ** 2 + 0.5 ** 2)
        phi = np.arctan2(dy, dx)
        tangential_shear = lenses.eR / (2 * r)
        self.gamma1 -= np.sum(tangential_shear * np.cos(2 * phi), axis=1)
        self.gamma2 -= np.sum(tangential_shear * np.sin(2 * phi), axis=1)


    def calc_flex(self, lenses):
        dx = self.x[:, np.newaxis] - lenses.x
        dy = self.y[:, np.newaxis] - lenses.y
        r2 = dx ** 2 + dy ** 2 + 0.5 ** 2
        phi = np.arctan2(-dy, dx)
        radial_flexion = lenses.eR / (2 * r2)
        self.f1 -= np.sum(radial_flexion * np.cos(phi), axis=1)
        self.f2 -= np.sum(radial_flexion * np.sin(phi), axis=1)


    def weights(self, size, sigma_f = 10**-2, sigma_g = 10**-3, eRmin = 1, eRmax = 60):
        # size: size of the grid in arcseconds
        # sigma_f: standard deviation of the flexion
        # sigma_g: standard deviation of the shear
        # eRmin: minimum allowed value of eR
        # eRmax: maximum allowed value of eR
        res = size 
        line = np.linspace(-size,size,res)
        x,y = np.meshgrid(line,line)
        xs, ys, F1, F2, gamma1, gamma2 = self.x, self.y, self.f1, self.f2, self.gamma1, self.gamma2
        F = np.sqrt(F1**2 + F2**2)
        phiF = np.arctan2(F2,F1) + np.pi
        gamma = np.sqrt(gamma1**2 + gamma2**2)
        phi_gamma = np.arctan2(gamma2,gamma1) / 2 + np.pi
        weights1 = np.ones((res,res,res))
        weights2 = np.ones((res,res,res))
        eR_range = np.linspace(eRmin + 0.2, eRmax,res)

        for n in range(len(xs)):
            xn = x - xs[n]
            yn = y + ys[n] # The y-axis is flipped
            r = np.sqrt(xn**2 + yn**2)
            phi1 = np.arctan2(-yn,xn) - phi_gamma[n]
            phi2 = np.arctan2(-yn,xn) + phiF[n] + np.pi/2

            shear_contribution = compute_weights(gamma[n], 'shear', r, phi1, eR_range, res, sigma_g, eRmin, eRmax)
            flexion_contribution = compute_weights(F[n], 'flexion', r, phi2, eR_range, res, sigma_f, eRmin, eRmax)

            if np.sum(shear_contribution) > 0:
                #Multiply the weights by the contribution from the nth lens and normalize
                weights1 *= shear_contribution
                weights1 /= np.sum(weights1)

            if np.sum(flexion_contribution) > 0:
                #Multiply the weights by the contribution from the nth lens and normalize
                weights2 *= flexion_contribution
                weights2 /= np.sum(weights2)
            
        weights3 = weights1 * weights2 # Combine the weights from the shear and flexion
        weights3 /= np.sum(weights3)

        return weights1, weights2, weights3


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
    lens_F = -(eR * np.abs(np.cos(phi))) / (2 * r**2)
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
        filter = np.exp(-r / 10)
        coefficient = 2 * filter * r**2 / np.abs(np.cos(phi)) 
    elif signal_type == 'shear':
        integrand = shear_integrand
        coefficient = 2 * r / np.abs(np.cos(2 * phi))

    for i in range(res):
        for j in range(res):
            denominator[i, j] = integrate.quad(integrand, eRmin, eRmax, args=(signal, r[i, j], sigma, phi[i, j]))[0]
            denominator += 1e-10  # Prevent divide by zero errors

    numerator = integrand(eR[:, None, None], signal, r, sigma, phi)
    unnormalized_weights = coefficient * numerator / denominator + 10 ** -10  # Prevent divide by zero errors

    weights = np.where(np.sum(unnormalized_weights) == 0, np.ones((res,res,res)), unnormalized_weights / np.sum(unnormalized_weights))

    return weights


def score_map(colormap, threshold=0.1):
    '''
    This function takes in a colormap and returns the locations of the peaks in the colormap
    as well as the scores associated with each peak. It also takes in a threshold value, which
    is used to determine the minimum score threshold. The minimum score threshold is defined as
    the maximum score in the colormap multiplied by the threshold value. The function returns
    the locations of the peaks and the scores associated with each peak that are above the minimum
    score threshold.
    '''

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

    return xloc, yloc, scores
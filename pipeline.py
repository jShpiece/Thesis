import numpy as np
from utils import chi2, chi2wrapper
import scipy.optimize as opt


def get_initial_guess(x, y, e1, e2, f1, f2):
    phi = np.arctan2(f2, f1)
    gamma = np.sqrt(e1**2 + e2**2)
    flexion = np.sqrt(f1**2 + f2**2)
    r = gamma / flexion
    return x + r * np.cos(phi), y + r * np.sin(phi), 2 * gamma * np.abs(r)


def find_initial_lens_positions(x, y, e1, e2, f1, f2, sigs, sigf):
    ns = len(x)
    xlens, ylens, te_lens = np.zeros(ns), np.zeros(ns), np.zeros(ns)
    for i in range(ns):
        params = [[x[i]], [y[i]], [e1[i]], [e2[i]], [f1[i]], [f2[i]], sigf, sigs]
        guess = get_initial_guess(x[i], y[i], e1[i], e2[i], f1[i], f2[i])
        result = opt.minimize(chi2wrapper, guess, args=(params))
        xlens[i], ylens[i], te_lens[i] = result.x
    return xlens, ylens, te_lens


def filter_lens_positions(xl, yl, eR, x, y, xmax, threshold_distance=1):
    # Filter out lenses that are too close to sources or too far from the center
    distances_to_sources = np.sqrt((xl[:, None] - x)**2 + (yl[:, None] - y)**2)
    too_close_to_sources = (distances_to_sources < threshold_distance).any(axis=1)
    too_far_from_center = np.sqrt(xl**2 + yl**2) > 2 * xmax

    valid_indices = ~(too_close_to_sources | too_far_from_center)
    return xl[valid_indices], yl[valid_indices], eR[valid_indices]


def merge_close_lenses(xlens, ylens, telens, merger_threshold=1):
    #Merge lenses that are too close to each other
    i = 0
    while i < len(xlens):
        for j in range(i+1, len(xlens)):
            distance = np.sqrt((xlens[i] - xlens[j])**2 + (ylens[i] - ylens[j])**2)
            if distance < merger_threshold:
                weight_i, weight_j = telens[i], telens[j]
                xlens[i] = (xlens[i]*weight_i + xlens[j]*weight_j) / (weight_i + weight_j)
                ylens[i] = (ylens[i]*weight_i + ylens[j]*weight_j) / (weight_i + weight_j)
                telens[i] = (weight_i + weight_j) / 2
                xlens, ylens, telens = np.delete(xlens, j), np.delete(ylens, j), np.delete(telens, j)
                break
        else:
            i += 1
    return xlens, ylens, telens


def iterative_elimination(xlens, ylens, telens, chi2val, x, y, e1, e2, f1, f2, sigf, sigs, lens_floor=1):
    # Iteratively eliminate lenses that do not improve the chi^2 value
    while True:
        if len(xlens) <= lens_floor:
            # If we have reached the minimum number of lenses, stop
            break
        best_indices = None
        for i in range(len(xlens)):
            test_xlens = np.delete(xlens, i)
            test_ylens = np.delete(ylens, i)
            test_telens = np.delete(telens, i)
            dof = 4 * len(x) - 3 * len(test_xlens)
            new_chi2val = chi2(x, y, e1, e2, f1, f2, test_xlens, test_ylens, test_telens, sigf, sigs) / dof
            if new_chi2val < chi2val:
                chi2val, best_indices = new_chi2val, i
                break
        if best_indices is None:
            break
        xlens, ylens, telens = np.delete(xlens, best_indices), np.delete(ylens, best_indices), np.delete(telens, best_indices)
    return xlens, ylens, telens, chi2val


def print_step_info(flags,message,x,xlens,chi2val):
    if flags:
        print(message)
        print('Number of lenses: ', len(xlens))
        print('Number of sources: ', len(x))
        if chi2val is not None:
            print('Chi^2: ', chi2val)


def get_chi2_value(x, y, e1, e2, f1, f2, xlens, ylens, telens, sigf, sigs):
    dof = 4 * len(x) - 3 * len(xlens)
    return chi2(x, y, e1, e2, f1, f2, xlens, ylens, telens, sigf, sigs) / dof


def optimize_lens_positions(x,y,e1data,e2data,f1data,f2data,sigs,sigf,xmax,flags = False):
    # Given a lensing configuration, find the optimal lens positions

    # Find initial lens positions with a local minimization
    xlens, ylens, telens = find_initial_lens_positions(x, y, e1data, e2data, f1data, f2data, sigs, sigf)
    chi2val = get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, telens, sigf, sigs)
    print_step_info(flags, "Initial chi^2:", x, xlens, chi2val)
    
    # Filter out lenses that are too close to sources or too far from the center
    xlens, ylens, telens = filter_lens_positions(xlens, ylens, telens, x, y, xmax)
    chi2val = get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, telens, sigf, sigs)
    print_step_info(flags, "After winnowing:", x, xlens, chi2val)

    # Merge lenses that are too close to each other
    xlens, ylens, telens = merge_close_lenses(xlens, ylens, telens)
    chi2val = get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, telens, sigf, sigs)
    print_step_info(flags, "After merging:", x, xlens, chi2val)
    
    # Iteratively eliminate lenses that do not improve the chi^2 value
    xlens, ylens, telens, chi2val = iterative_elimination(xlens, ylens, telens, chi2val, x, y, e1data, e2data, f1data, f2data, sigf, sigs)
    print_step_info(flags, "After iterative minimization:", x, xlens, chi2val)

    return xlens, ylens, telens, chi2val
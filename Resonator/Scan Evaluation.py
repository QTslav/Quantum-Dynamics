import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import fftpack
from matplotlib.widgets import SpanSelector
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import random

def onselect1(xmin, xmax):
    for i, line in enumerate(axs[1].get_lines()):
        x = line.get_xdata()
        x_selected = x[np.where( (x>=xmin) & (x<=xmax))[0]]
        y = line.get_ydata()    
        y_selected = y[np.where( (x>=xmin) & (x<=xmax))[0]]
        if np.sum(y_selected) != 0:
            print(i, x_selected@y_selected/np.sum(y_selected))

def onselect2(xmin, xmax):
    for i, line in enumerate(axs[3].get_lines()):
        x = line.get_xdata()
        x_selected = x[np.where( (x>=xmin) & (x<=xmax))[0]]
        y = line.get_ydata()    
        y_selected = y[np.where( (x>=xmin) & (x<=xmax))[0]]
        if np.sum(y_selected) != 0:
            print(i, x_selected@y_selected/np.sum(y_selected))

p_mode_fits = [[],[],[]]
def onselect3(xmin, xmax, p_mode_fit=p_mode_fits):
    for i, line in enumerate(axs[5].get_lines()):
        x = line.get_xdata()
        x_selected = x[np.where( (x>=xmin) & (x<=xmax))[0]]
        y = line.get_ydata()    
        y_selected = y[np.where( (x>=xmin) & (x<=xmax))[0]]
        if np.sum(y_selected) != 0 and line.get_color() in ['k', 'orange'] and len(x_selected)>1:
            # print(f"Line {i} - x @ ymax", x_selected[np.argmax(y_selected)])
            p_mode_fits[0].append(x_selected[np.argmax(y_selected)])
            mu_avg = x_selected@y_selected/np.sum(y_selected)
            # print(f"Line {i} - x averaged {x_selected@y_selected/np.sum(y_selected)}")
            p_mode_fits[1].append(mu_avg)
            try:
                p0 = (mu_avg, x_selected[1]-x_selected[0], np.amax(y_selected), 0)
                bounds = ([x_selected[0], (x_selected[1]-x_selected[0])/10, 0, 0], [x_selected[-1], x_selected[-1]-x_selected[0], np.sum(y_selected), 1])
                (mu, sigma, I, c), pcov = curve_fit(lorentzian, x_selected, y_selected, p0=p0, bounds=bounds)
                p_mode_fits[2].append(mu)
                print(f"Line {i} - Lorentzian Fit: mu {mu}, sigma {sigma}, I {I}")  
                x_fit = np.linspace(x_selected[0], x_selected[-1],100)
                axs[5].plot(x_fit, lorentzian(x_fit, mu, sigma, I, c), '-r', alpha=0.5, linewidth=1)
            except Exception as e:
                print("Could not fit Lorentzian due to ", e)
    print(len(p_mode_fit[0]))
    if len(p_mode_fit[0])>0 and len(p_mode_fit[0])%4==0:
        
        for p_mode_fit in p_mode_fits:
            print(50*"-")
            print("Fitting ", p_mode_fit)
            p_mode_fit = np.array(p_mode_fit)
            p_mode_fit = np.sort(p_mode_fit)
            mode_idx = (p_mode_fit[1]-p_mode_fit[0])/((p_mode_fit[3]-p_mode_fit[1])-(p_mode_fit[2]-p_mode_fit[0]))
            print("Mode index: ", mode_idx)
            print("Cavity length: ", mode_idx*737e-3/2)
            print(50*"-")


def normalize(V):
    return (V-np.amin(V))/np.amax((V-np.amin(V)))


def lorentzian(lbd, lbd0, Gamma, I, c):
    return I * 1/np.pi * 1/2*Gamma/( (lbd-lbd0)**2 + (1/2*Gamma)**2 ) + c


def gaussian(x, mu, sigma, I):
    return I*1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))


def estimate_lorentzian(t, V):
    
    mu_est = np.argmax(V)
    Gamma_est = t[-1]-t[0]
    I_est = 2/(Gamma_est*np.pi)
    c_est = 0
    
    p0 = [mu_est, Gamma_est, I_est, c_est]
    bounds = ([t[0], 0, 0, 0], [t[-1],  2*Gamma_est, 2*I_est, 0.1])
    popt, pcov = curve_fit(lorentzian, t, V, p0=p0, bounds=bounds)
    
    try:
        popt, pcov = curve_fit(lorentzian, t, V, p0=p0, bounds=bounds)
        lbd0, Gamma, I, c = popt
        return lbd0, Gamma, I, c

    except RuntimeError as re:
        print("To few data points: ", re)
        return None

    except ValueError as ve:
        print("Value error: ", ve)
        return None
    

def high_pass(t, V, nu_cutoff):
    N = len(V)
    nu = np.fft.fftfreq(N,t[1]-t[0])
    fft = np.fft.fft(V)
    fft[np.where(np.abs(nu)<nu_cutoff)[0]] = 0
    V = np.fft.ifft(fft)
    return np.real(V)


eps = 0.05
# Pick cavity scan data from bichromatic scans
Tk().withdraw() 
stop_reading = False
filenames = []
scans = []
while True:
    filename = askopenfilename()
    if filename and filename != '':
        filenames.append(filename)
    else:
        break

tmp_up = []
tmp_down = []
for filename in filenames:
    print(filename)
    scans.append(np.genfromtxt(open(filename, "rb"), delimiter="\t"))
    tmp_up.append([])
    tmp_down.append([])

scan_colors = ['b', 'g', 'r', 'm', 'k', 'orange']
verbose = False
num_plots = 2
fig, axs = plt.subplots(2*len(filenames),1, figsize=(20,10))

for a, scan in enumerate(scans):
    # Open counts from photodiode with format Piezo Voltage (V), Diode Voltage (V/bin), Voltage/bin
    t = scan[:,2]
    V_Piezo = scan[:,0]
    V_Diode = scan[:,1]
    
    axs[0+a*num_plots].plot(t, V_Diode, '-k', alpha=0.8)
    axs[0+a*num_plots].set_title("Original Signal")
    axs[0+a*num_plots].set_xlabel("time (s)")
    axs[0+a*num_plots].set_ylabel("Diode Voltage (V)")
    axs01 = axs[0+a*num_plots].twinx()
    axs01.plot(t, V_Piezo, '-.b', alpha=0.5)
    axs01.set_ylabel("Piezo Voltage")


    # Split ramps by looking for voltage minima and maxima in piezo voltage
    ramp_min_idxs = (np.where(V_Piezo == np.amin(V_Piezo))[0]).astype(int)
    ramp_max_idxs = (np.where(V_Piezo == np.amax(V_Piezo))[0]).astype(int)

    # Filter out duplicate ramp minima
    temp = [ramp_min_idxs[0]]
    for i in range(1,len(ramp_min_idxs)):
        if (ramp_min_idxs[i]-ramp_min_idxs[i-1])>1:
            temp.append(ramp_min_idxs[i])
        else:
            continue    
    ramp_min_idxs = temp

    # Filter out duplicate ramp maxima
    temp = [ramp_max_idxs[0]]
    for i in range(1,len(ramp_max_idxs)):
        if (ramp_max_idxs[i]-ramp_max_idxs[i-1])>1:
            temp.append(ramp_max_idxs[i])
        else:
            continue    
    ramp_max_idxs = temp


    # Calculate ramp indices by discriminating between starting downward and upward ramp,
    # which leads to either more maxima or minima in piezo voltage
    ramps = []
    if len(ramp_min_idxs) > len(ramp_max_idxs):
        for i in range( min( len(ramp_min_idxs), len(ramp_max_idxs) ) ):
            ramps.append([ramp_min_idxs[i], ramp_max_idxs[i]])
            ramps.append([ramp_max_idxs[i], ramp_min_idxs[i+1]])
    elif len(ramp_min_idxs) == len(ramp_max_idxs):
        for i in range(len(ramp_min_idxs)):
            ramps.append([ramp_min_idxs[i], ramp_max_idxs[i]])
            if i<len(ramp_min_idxs)-1:
                ramps.append([ramp_max_idxs[i], ramp_min_idxs[i+1]])
    else:
        for i in range( min( len(ramp_min_idxs), len(ramp_max_idxs) ) ):
            ramps.append([ramp_max_idxs[i], ramp_min_idxs[i]])
            ramps.append([ramp_min_idxs[i], ramp_max_idxs[i+1]])

    # Threshold in percent of maximum resonance
    threshold = 0.4
    num_peak_pts = 10
    fundamental_modes_fit = []
    fundamental_modes_idx = []
    fsr = []
    t_ramps = []
    V_Diode_ramps_norm = []
    for i, ramp in enumerate(ramps):
        if verbose: 
            print(50*"-")    


        # ------------------------ Data Preparation ------------------------
        #Truncate ramp to stay away from inflection points
        ramp = np.arange(ramp[0] + int((eps)*(ramp[1]-ramp[0])) , ramp[0] + int((1-eps)*(ramp[1]-ramp[0])) )
        t_ramp = t[ramp]
        t_ramps.append(t_ramp)
        V_Piezo_ramp = V_Piezo[ramp]
        V_Diode_ramp = V_Diode[ramp]
        
        # Normalize to range of 0 and 1
        V_Diode_ramp_norm = normalize(V_Diode_ramp)
        V_Diode_ramp_norm = 1 - V_Diode_ramp_norm
        
        # Apply highpass filter with cutoff of 1000Hz for laser polarization drifts
        # V_Diode_ramp_norm = high_pass(t_ramp, V_Diode_ramp_norm, 1e3)
        
        # Assume remaining nosie to be gaussian white noise and fit a gaussian to the histogram of counts
        # Then set a threshold for which values are set to zero
        # hist, bins = np.histogram(V_Diode_ramp_norm, bins=100)
        # bins = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        # (mu, sigma, I), pcov = curve_fit(gaussian, bins, hist)
        # V_Diode_ramp_norm[np.where(V_Diode_ramp_norm<sigma+mu)[0]] = 0
        # V_Diode_ramp_norm = normalize(V_Diode_ramp_norm)
        
        V_Diode_ramps_norm.append(V_Diode_ramp_norm)
        
        axs[1+a*num_plots].plot(t_ramp, V_Diode_ramp_norm, color=scan_colors[2*a+(i%2)], linestyle='-', linewidth=1, alpha=1)
        if a !=len(scans)-1:
            axs[1+a*num_plots].plot(t_ramp, normalize(V_Diode_ramp), color=scan_colors[2*a+(i%2)], alpha=0.1, linewidth=0.5, linestyle='-.')
            axs[(len(scans)-1)*num_plots+1].plot(t_ramp, V_Diode_ramp_norm, color=scan_colors[2*a+(i%2)], linestyle='-', linewidth=1, alpha=0.4)
        
        
        # Plot ramps over eachother to see drift induced by piezo due to nonlinearities for example
        # axs[2+a*num_plots].plot(t_ramp-t_ramp[0], V_Diode_ramp_norm, linewidth=1, color=scan_colors[2*a+(i%2)], linestyle='-', alpha=0.5)


        # ------------------------ Peak Finder ------------------------
        # Filter out single peaks by looking for all elements greater than x (e.g. 0.5) 
        peaks_idxs = np.where(V_Diode_ramp_norm >= threshold*np.amax(V_Diode_ramp_norm))[0]
        
        # Split the resulting array of indices containing a peak by looking for discontinouities
        # in the index array which corresponds to the next peak
        peaks_maxs_ramp = []
        for peak_idxs in np.array_split( peaks_idxs, np.where(np.diff(peaks_idxs)>2)[0] + 1):
            axs[1+a*num_plots].plot(t_ramp[peak_idxs[np.argmax(V_Diode_ramp_norm[peak_idxs])]], V_Diode_ramp_norm[peak_idxs[np.argmax(V_Diode_ramp_norm[peak_idxs])]],
                        color=scan_colors[2*a+(i%2)], marker='+', alpha=0.8, linewidth=0.1)
            peaks_maxs_ramp.append(peak_idxs[np.argmax(V_Diode_ramp_norm[peak_idxs])])
        

        # ------------------------ FSR Detection ------------------------
        # In order to find the free spectral range, iterate every ramp from total peaks,
        # calculate difference of every peak to every other peak and look for maxima in histogram
        # spectral range should be periodic within each ramp.  
        num_peaks = 0
        peak_dist = []
        for j in range(len(peaks_maxs_ramp)):
            num_peaks+=1

            for k in range(len(peaks_maxs_ramp)):
                if k==j: continue
                peak_dist.append(np.abs(peaks_maxs_ramp[k]-peaks_maxs_ramp[j]))
        
        hist, bins = np.histogram(peak_dist, bins=5)
        bins = np.array([int((bins[l+1]+bins[l])/2) for l in range(len(bins)-1)])
        if bins[np.argmax(hist)] != 0:
            fsr.append(bins[np.argmax(hist)])
            if verbose: 
                print(f"Ramp {i}: FSR {t_ramp[fsr[-1]]-t_ramp[0]}")
        else: 
            print("Cannot determine FSR")
            print("Histogram:")
            print(hist)
            continue
        


        # ------------------------ Fundamental Mode Detection ------------------------
        # Initial fundamental mode to be defined as the highest peak within ramp
        fundamental_modes_idx_ramp = []  
        fundamental_mode_idx = peaks_maxs_ramp[np.argmax(V_Diode_ramp_norm[peaks_maxs_ramp])]
        axs[1+a*num_plots].plot(t_ramp[fundamental_mode_idx], V_Diode_ramp_norm[fundamental_mode_idx], color='r', marker='o', alpha=1)
        # axs[3+a*num_plots].plot(t_ramp[fundamental_mode_idx]-t_ramp[0], V_Diode_ramp_norm[fundamental_mode_idx], color='r', marker='+', alpha=0.5)
        fundamental_modes_idx_ramp.append(fundamental_mode_idx)
        
        if i%2 == 0:
            tmp_up[a].append(t_ramp[fundamental_mode_idx]-t_ramp[0])
        else:
            tmp_down[a].append(t_ramp[fundamental_mode_idx]-t_ramp[0])

        # Try to fit a lorentzian to the proposed fundamental mode
        peak_idxs = np.arange(fundamental_mode_idx-int(num_peak_pts/2),fundamental_mode_idx+int(num_peak_pts/2))
        V_Diode_peak = V_Diode_ramp_norm[fundamental_mode_idx-int(num_peak_pts/2):fundamental_mode_idx+int(num_peak_pts/2)]
        fundamental_modes_fit_ramp = []
        try:
            popt, pcov = curve_fit(lorentzian, peak_idxs, V_Diode_peak, 
                                    p0 = [fundamental_mode_idx, 1, 1, 0],
                                    bounds = ( [np.amin(peak_idxs), 0, 0, 0], [np.amax(peak_idxs), np.amax(peak_idxs)-np.amin(peak_idxs), 1e3, 0.1]))
            (lbd0, Gamma, I, c) = popt
            fundamental_modes_fit_ramp.append([lbd0, Gamma, I, c])  
            
            # Plot fit in seperate plot
            peak_x = np.linspace(peak_idxs[0], peak_idxs[-1],100)

            if verbose: 
                print(f"Ramp {i}: Finesse 0 {fsr[-1]/Gamma}")

        except Exception as e:
            fundamental_modes_fit_ramp.append(e)
        
        # In order to look for other fundamental modes, reduce FSR to catch peaks which might lay
        # outside of the fsr, determined above.
        fsr_tmp = int(0.95*fsr[-1])
        for m in range(len(t_ramp)//fsr_tmp +1):

            # Subtract the number of fsrs from the first guess, i.e. the largest peak, to get the next guess
            guess = fundamental_mode_idx-fsr_tmp*(fundamental_mode_idx // fsr_tmp) + m*fsr_tmp
            
            # Look for differences of peak maxima within one ramp to the proposed guess and take the minimum
            # as the next higher mode
            fundamental_mode_idx = peaks_maxs_ramp[np.argmin(np.abs(peaks_maxs_ramp-guess))]
            axs[1+a*num_plots].plot(t_ramp[fundamental_mode_idx], V_Diode_ramp_norm[fundamental_mode_idx], color='r', marker='o', alpha=1)
            # axs[3+a*num_plots].plot(t_ramp[fundamental_mode_idx]-t_ramp[0], V_Diode_ramp_norm[fundamental_mode_idx], color='r', marker='+', alpha=0.5)
            fundamental_modes_idx_ramp.append(fundamental_mode_idx)

            if i%2 == 0:
                tmp_up[a].append(t_ramp[fundamental_mode_idx]-t_ramp[0])
            else:
                tmp_down[a].append(t_ramp[fundamental_mode_idx]-t_ramp[0])

            # Repeat procedure for other fundamental modes as well
            peak_idxs = np.arange(fundamental_mode_idx-int(num_peak_pts/2),fundamental_mode_idx+int(num_peak_pts/2))
            V_Diode_peak = V_Diode_ramp_norm[fundamental_mode_idx-int(num_peak_pts/2):fundamental_mode_idx+int(num_peak_pts/2)]

            try:
                popt, pcov = curve_fit(lorentzian, peak_idxs, V_Diode_peak, 
                                        p0 = [fundamental_mode_idx, 1, 1, 0],
                                        bounds = ( [np.amin(peak_idxs), 0, 0, 0], [np.amax(peak_idxs), np.amax(peak_idxs)-np.amin(peak_idxs), 1e3, 0.1]))
                (lbd0, Gamma, I, c) = popt
                fundamental_modes_fit_ramp.append([lbd0, Gamma, I, c])  

                # Plot fit in seperate plot
                peak_x = np.linspace(peak_idxs[0], peak_idxs[-1],100)

                if verbose: 
                    print(f"Ramp {i}: Finesse {m+1} {fsr[-1]/Gamma}")

            except Exception as e:
                fundamental_modes_fit_ramp.append(e)  

        # Collect all fundamental modes of the currenct ramp
        fundamental_modes_idx.append(fundamental_modes_idx_ramp)
        fundamental_modes_fit.append(fundamental_modes_fit_ramp)
        if verbose: 
            print(50*"-")    


if len(tmp_up[0])>0:
    # hist_up, bin_edges_up = np.histogram(np.array(tmp_up[0]), bins=num_plots*len(tmp_up[0]))
    # bins_up = [(bin_edges_up[i]+bin_edges_up[i+1])/2 for i in range(len(bin_edges_up)-1)]
    # axs[1].plot(bins_up, hist_up, color=scan_colors[0])
    # hist_down, bin_edges_down = np.histogram(np.array(tmp_down[0]), bins=num_plots*len(tmp_down[0]))
    # bins_down = [(bin_edges_down[i]+bin_edges_down[i+1])/2 for i in range(len(bin_edges_down)-1)]
    # axs[1].plot(bins_down, hist_down, color=scan_colors[2])

    span = SpanSelector(axs[1], onselect1, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
# if len(tmp_up[1])>0:
    # hist_up, bin_edges_up = np.histogram(np.array(tmp_up[1]), bins=num_plots*len(tmp_up[1]))
    # bins_up = [(bin_edges_up[i]+bin_edges_up[i+1])/2 for i in range(len(bin_edges_up)-1)]
    # axs[3].plot(bins_up, hist_up, color=scan_colors[1], marker='.')
    # hist_down, bin_edges_down = np.histogram(np.array(tmp_down[1]), bins=num_plots*len(tmp_down[1]))
    # bins_down = [(bin_edges_down[i]+bin_edges_down[i+1])/2 for i in range(len(bin_edges_down)-1)]
    # axs[3].plot(bins_down, hist_down, color=scan_colors[3], marker='.')

    # span1 = SpanSelector(axs[3], onselect2, 'horizontal', useblit=True,
    #                 rectprops=dict(alpha=0.5, facecolor='red'))

# if len(tmp_up[2])>0:
    # hist_up, bin_edges_up = np.histogram(np.array(tmp_up[2]), bins=num_plots*len(tmp_up[2]))
    # bins_up = [(bin_edges_up[i]+bin_edges_up[i+1])/2 for i in range(len(bin_edges_up)-1)]
    # axs[5].plot(bins_up, hist_up, color=scan_colors[2], marker='.')
    # hist_down, bin_edges_down = np.histogram(np.array(tmp_down[2]), bins=num_plots*len(tmp_down[2]))
    # bins_down = [(bin_edges_down[i]+bin_edges_down[i+1])/2 for i in range(len(bin_edges_down)-1)]
    # axs[5].plot(bins_down, hist_down, color=scan_colors[5], marker='.')

    # span1 = SpanSelector(axs[5], onselect3, 'horizontal', useblit=True,
    #                 rectprops=dict(alpha=0.5, facecolor='red'))


plt.tight_layout()
plt.show()
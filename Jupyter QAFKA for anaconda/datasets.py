import numpy as np
import tiffcapture as tc
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt
from utils import *

def CreateMaxDataSet(data, max_size, seg):
    # Divide the Data_Set into sum_size datasets summing every frame of data in each segment
    max_data = np.zeros([max_size, data.shape[1], data.shape[2]])
    divider = int(data.shape[0]/max_size)
    n = divider - seg
    max_data[0, :, :] = np.max(data[:n, :, :], axis=0)
    n = divider
    for i in range(1, max_size):
        max_data[i, :, :] = np.max(data[((i-1)*n):(i*n), :, :], axis=0)
    return max_data

def LocalizeEmitters(data, sum_threshold, quality_threshold, pixel_length, resolution_nm, emitters_size):
    total_frames, max_row, max_col = data.shape
    emitters_grid = np.zeros((int(max_row * pixel_length / resolution_nm),
                              int(max_col * pixel_length / resolution_nm)), dtype=int)
    emitters_coordinates = []
    emitters_cnt = 0

    patch_length = 9  # Determine the patch we are going to fit to be of size [patch_length x patch length]
    xy = np.zeros([2, int(patch_length ** 2)])
    for i1 in range(patch_length):
        for j1 in range(patch_length):
            xy[:, int(i1 * patch_length + j1)] = [i1, j1]

    # Create the shape of a circle for the emitters grid
    circle = draw_circle(size=int(patch_length * pixel_length / resolution_nm),
                         rad=int(emitters_size / resolution_nm))

    low_intensity, bad_fit, out_of_bound = 0, 0, 0
    for frame in range(total_frames):
        img_arr = data[frame, :, :]
        threshold = sum_threshold[frame]
        if (np.max(img_arr) > threshold):
            potential_peaks = np.where(img_arr > threshold)
            for i in range(len(potential_peaks[0])):
                row, col = potential_peaks[0][i], potential_peaks[1][i]

                # Handle clusters in case of exceeding image shape
                up = int(row + np.floor(patch_length / 2)) + 1
                down = int(row - np.floor(patch_length / 2))
                left = int(col - np.floor(patch_length / 2))
                right = int(col + np.floor(patch_length / 2)) + 1

                # Ignore out of bound blinks
                if up > max_row or down < 0 or left < 0 or right > max_col:
                    out_of_bound += 1
                    continue

                # Initial guess
                x0, y0 = int(np.floor(patch_length / 2)), int(np.floor(patch_length / 2))

                # Fit the patch to a gaussian
                zobs = (img_arr[down:up, left:right]).reshape(1, -1).squeeze()

                # Check if localization is in local maximum
                if ([down + int(np.argmax(zobs)/patch_length), left + np.argmax(zobs) % patch_length] != [row, col]):
                    continue

                guess = [np.median(img_arr), np.max(img_arr) - np.min(img_arr), x0, y0, 1]
                try:
                    pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)
                except:
                    continue

                fit = gauss2d(xy, *pred_params)

                # If the peak is higher than threshold proceed
                curr_row = int(np.round(down + pred_params[3]))
                curr_col = int(np.round(left + pred_params[2]))
                if (curr_col >= max_col or curr_col < 0 or curr_row >= max_row or curr_row < 0):
                    out_of_bound += 1
                    continue

                # Handle same coordinate repetition
                y = down + pred_params[3]
                x = left + pred_params[2]
                center_y = int(np.round(y * pixel_length / resolution_nm))
                center_x = int(np.round(x * pixel_length / resolution_nm))

                if (img_arr[curr_row, curr_col] > threshold):
                    # Calculate RMS
                    zobs /= np.max(zobs)
                    fit /= np.max(fit)
                    fit_quality = 1 - np.sqrt(np.mean((zobs - fit) ** 2))

                    # If the fit quality is higher than Value > take the mean value as a new cluster's coordinates
                    # Ignore fitted gaussian with sigma higher than 1 or lower than 0.3
                    #if (fit_quality > quality_threshold and pred_params[4] < 1 and pred_params[4] > 0.3):
                    if (fit_quality > quality_threshold):
                        # If the current pixel in the grid is already tagged for one of the emitters
                        if (emitters_grid[center_y, center_x] > 0):
                            continue

                        mid = int((patch_length * pixel_length / resolution_nm)/2)
                        # If the emitter is located outside the image boundaries ignore
                        if(center_y - mid < 0 or center_y + mid + 1 >= emitters_grid.shape[0] or
                                center_x - mid < 0 or center_x + mid + 1 >= emitters_grid.shape[1]):
                            continue

                        # Add localization
                        emitters_cnt += 1
                        # Update emitters_grid
                        emitters_grid[(center_y-mid):(center_y+mid+1), (center_x-mid):(center_x+mid+1)] += emitters_cnt * circle
                        # Update emitters list
                        emitters_coordinates.append([center_y, center_x])
                    else:
                        bad_fit += 1
                else:
                    low_intensity += 1

    print("Emitter is out of bound:", out_of_bound)
    print("Bad fitting grade:", bad_fit)
    print("Emitters intensity is too low:", low_intensity)
    print("-I- found", emitters_cnt, "emitters")

    return np.array(emitters_coordinates)

def ExtractTimeTraces(raw_data, emitters_coord, pixel_length, resolution_nm, quality_threshold, threshold, emitters_size):
    total_frames, max_row, max_col = raw_data.shape
    emitters = np.zeros([emitters_coord.shape[0], total_frames])
    emitters_size_pix = emitters_size / resolution_nm
    patch_length = 5 # Determine the patch we are going to fit to be of size [patch_length x patch length]
    xy = np.zeros([2, int(patch_length ** 2)])
    for i1 in range(patch_length):
        for j1 in range(patch_length):
            xy[:, int(i1 * patch_length + j1)] = [i1, j1]

    low_intensity, bad_fit, out_of_bound, far_from_emitters = 0, 0, 0, 0
    for frame in range(total_frames):
        img_arr = raw_data[frame, :, :]
        if (np.max(img_arr) > threshold):
            potential_peaks = np.where(img_arr > threshold)
            for i in range(len(potential_peaks[0])):
                row, col = potential_peaks[0][i], potential_peaks[1][i]

                # Handle clusters in case of exceeding image shape
                up = int(row + np.floor(patch_length / 2)) + 1
                down = int(row - np.floor(patch_length / 2))
                left = int(col - np.floor(patch_length / 2))
                right = int(col + np.floor(patch_length / 2)) + 1

                # Ignore out of bound blinks
                if up > max_row or down < 0 or left < 0 or right > max_col:
                    out_of_bound += 1
                    continue

                # Initial guess
                x0, y0 = int(np.floor(patch_length / 2)), int(np.floor(patch_length / 2))

                zobs = (img_arr[down:up, left:right]).reshape(1, -1).squeeze()
                guess = [np.median(img_arr), np.max(img_arr) - np.min(img_arr), x0, y0, 1]
                try:
                    pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)
                except:
                    continue

                fit = gauss2d(xy, *pred_params)

                # Check if localization is in local maximum
                if (np.argmax(zobs) != int(np.round(pred_params[2])) * patch_length + int(np.round(pred_params[3]))):
                    continue

                # If the peak is higher than threshold proceed
                curr_row = int(np.round(down + pred_params[3]))
                curr_col = int(np.round(left + pred_params[2]))
                if (curr_col >= max_col or curr_col < 0 or curr_row >= max_row or curr_row < 0):
                    out_of_bound += 1
                    continue

                # Handle same coordinate repetition
                x = left + pred_params[2]
                y = down + pred_params[3]
                center_y = int(np.round(y * (pixel_length / resolution_nm)))
                center_x = int(np.round(x * (pixel_length / resolution_nm)))

                if (img_arr[curr_row, curr_col] > threshold):
                    # Calculate RMS
                    zobs /= np.max(zobs)
                    fit /= np.max(fit)
                    fit_quality = 1 - np.sqrt(np.mean((zobs - fit) ** 2))

                    # If the fit quality is higher than Value > take the mean value as a new cluster's coordinates
                    if (fit_quality > quality_threshold):
                        min_dist = np.inf
                        cluster_ind = -1
                        for k in range(emitters_coord.shape[0]):
                            dist = np.sqrt((center_y - emitters_coord[k, 0])**2 + (center_x - emitters_coord[k, 1])**2)
                            if (dist > emitters_size_pix):
                                continue
                            if (dist < min_dist):
                                min_dist = dist
                                cluster_ind = k
                        if (min_dist > emitters_size_pix):
                            far_from_emitters += 1
                            continue
                        else:
                            # Update emitters list with new time trace
                            emitters[cluster_ind, frame] = pred_params[0] + pred_params[1]  # Offset + Amp of gaussian
                    else:
                        bad_fit += 1
                else:
                    low_intensity += 1

    print("Blink too far from emitters:", far_from_emitters)
    print("Emitter is out of bound:", out_of_bound)
    print("Bad fitting grade:", bad_fit)
    print("Emitters intensity is too low:", low_intensity)
    print("-I- updated emitters time traces")

    return emitters

def CreateDataSet(file, chop):
    """
        Creates trajectories data set out of experimental data
        :param path: String for the path to the data library
        :param path: List for start frame and stop frame of the TIFF videos
        :return: data set [# of trajectories, trajectories length].
    """
    path = 'D:\Project\data'
    tiff = tc.opentiff(os.path.join(path, file))
    if chop[0] > tiff.length or chop[1] > tiff.length:
      print("Specified end_frame is bigger than the stack size")
      return -1
    x0 = tiff.find_and_read(chop[0])

    data = np.zeros((chop[1] - chop[0], x0.shape[0], x0.shape[1]))
    for i in range(chop[1] - chop[0]):
        data[i, :, :] = tiff.find_and_read(chop[0] + i)

    return data

def find_trajectories_gauss(raw_data, threshold, quality_threshold):
    """
        Finds trajectories from the given TIFF video
        :param raw_data: Tensor [# of trajectories, trajectories length]
        :param threshold: Float for the minimal threshold for peak detection
        :return: List of all the different trajectories in time.
    """
    total_frames, max_row, max_col = raw_data.shape
    resolution_nm = 20
    emitters_size = 8 # value * resolution_nm is the radius
    emitters_grid = np.zeros((int(max_row * 100 / resolution_nm), int(max_col * 100 / resolution_nm)), dtype=int)
    emitters_coordinates = []
    emitters = []
    emitters_cnt = 0

    patch_length = 9 # Determine the patch we are going to fit to be of size [patch_length x patch length]
    xy = np.zeros([2, int(patch_length ** 2)])
    for i1 in range(patch_length):
        for j1 in range(patch_length):
            xy[:, int(i1 * patch_length + j1)] = [i1, j1]

    # Fit patch to gaussian
    def gauss2d(xy, offset, amp, x0, y0, sigma):
        x, y = xy
        return offset + amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    for frame in range(total_frames):
        img_arr = raw_data[frame, :, :]
        if (np.max(img_arr) > threshold):
            potential_peaks = np.where(img_arr > threshold)
            for i in range(len(potential_peaks[0])):
                row, col = potential_peaks[0][i], potential_peaks[1][i]

                # Handle clusters in case of exceeding image shape
                up = int(row + np.floor(patch_length / 2)) + 1
                down = int(row - np.floor(patch_length / 2))
                left = int(col - np.floor(patch_length / 2))
                right = int(col + np.floor(patch_length / 2)) + 1

                # Ignore out of bound blinks
                if up > max_row or down < 0 or left < 0 or right > max_col:
                    continue

                # Ignore 1 pixel fake blinks
                pixel_cnt = np.where(img_arr[down:up, left:right] > threshold)[0]
                if (len(pixel_cnt) < 2):
                    continue

                # Initial guess
                x0, y0 = int(np.floor(patch_length/2)), int(np.floor(patch_length/2))

                zobs = (img_arr[down:up, left:right]).reshape(1, -1).squeeze()
                guess = [np.median(img_arr), np.max(img_arr) - np.min(img_arr), x0, y0, 1]
                try:
                    pred_params, uncert_cov = opt.curve_fit(gauss2d, xy, zobs, p0=guess)
                except:
                    continue

                fit = gauss2d(xy, *pred_params)

                # Check if the gaussian's variance is too small or too big
                if(pred_params[4] < 0.5 or pred_params[4] > 1.5):
                    continue

                # Taken from H. Flyvbjerg, et al. (2010) "Optimized localization analysis for single-molecule tracking and super-resolution microscopy"
                '''sigma = pred_params[4] # Fitted sigma of the gaussian
                a_squared = 1 # Pixel size
                b_squared = 100/(patch_length ** 2) # Background's expected photons per pixel
                N = 1 # Number of photons in a spot
                sigma_a_squared = (sigma ** 2) + a_squared / 12
                Variance = (sigma_a_squared / N) * (16/9 + (8*np.pi*sigma_a_squared*b_squared)/(N*a_squared))'''

                # If the peak is higher than threshold proceed
                curr_row = int(down + pred_params[3])
                curr_col = int(left + pred_params[2])
                if(curr_col > max_col or curr_col < 0 or curr_row > max_row or curr_row < 0):
                    continue

                # Handle same coordinate repetition
                x = left + pred_params[2]
                y = down + pred_params[3]
                center_y = int(y * 100 / resolution_nm)
                center_x = int(x * 100 / resolution_nm)

                if (img_arr[curr_row, curr_col] > threshold):
                    # Calculate RMS
                    zobs /= np.max(zobs)
                    fit /= np.max(fit)
                    fit_quality = 1 - np.sqrt(np.mean((zobs - fit) ** 2))

                    # If the fit quality is higher than Value > take the mean value as a new cluster's coordinates
                    if (fit_quality > quality_threshold):

                        # If the current pixel in the grid is already tagged for one of the emitters
                        if(emitters_grid[center_y, center_x] > 0):
                            # If the current blink has already been discovered
                            if (emitters[emitters_grid[center_y, center_x] - 1][frame] > 0):
                                continue
                            else:
                                # Update emitters_grid
                                emitters_grid[center_y - emitters_size:center_y + emitters_size,
                                              center_x - emitters_size:center_x + emitters_size] = emitters_grid[center_y, center_x]
                                # Update emitters list
                                emitters[emitters_grid[center_y, center_x] - 1][frame] = pred_params[0] + pred_params[1]
                                continue

                        # Add localization
                        emitters_cnt += 1
                        # Update emitters_grid
                        emitters_grid[center_y-emitters_size:center_y+emitters_size,
                                      center_x-emitters_size:center_x+emitters_size] = emitters_cnt
                        # Update emitters list
                        emitters_coordinates.append([y, x])
                        time_trace = np.zeros(total_frames)
                        time_trace[frame] = pred_params[0] + pred_params[1]# Offset + Amp of gaussian
                        emitters.append(time_trace)

    print("-I- found", len(emitters), "emitters")

    return np.array(emitters), np.array(emitters_coordinates)

def find_trajectories(raw_data, threshold, MaxTraceLength):
    """
        Finds trajectories from the given TIFF video
        :param raw_data: Tensor [# of trajectories, trajectories length]
        :param threshold: Float for the minimal threshold for peak detection
        :return: List of all the different trajectories in time.
    """
    dict_vec = []
    dist = 5
    total_frames, max_row, max_col = raw_data.shape

    for frame in range(total_frames):
        img_arr = raw_data[frame, :, :]
        if (np.max(img_arr) > threshold):
            potential_peaks = np.where(img_arr > threshold)

            for i in range(len(potential_peaks[0])):
                row, col = potential_peaks[0][i], potential_peaks[1][i]
                up = row + 1 if row < max_row else row
                down = row - 1 if row > 0 else row
                left = col - 1 if col > 0 else col
                right = col + 1 if col < max_col else col

                if (np.mean(img_arr[down:up+1, left:right+1]) > threshold):
                    flag = False
                    for k in range(len(dict_vec)):
                        if (row == dict_vec[k]['row'] and col == dict_vec[k]['col'] and
                                np.abs(frame - dict_vec[k]['start']) < MaxTraceLength):
                            flag = True
                            break
                        else:
                            if(np.abs(row - dict_vec[k]['row']) < dist and np.abs(col - dict_vec[k]['col']) < dist and
                                    np.abs(frame - dict_vec[k]['start']) < MaxTraceLength):
                                start = dict_vec[k]['start'] if dict_vec[k]['start'] < total_frames else total_frames - MaxTraceLength
                                rowK = dict_vec[k]['row']
                                colK = dict_vec[k]['col']
                                trace1 = raw_data[start:(start+MaxTraceLength), row, col]
                                trace2 = raw_data[start:(start+MaxTraceLength), rowK, colK]
                                corr = np.corrcoef(trace1, trace2)[0, 1]
                                if(np.abs(corr) > 0.1):
                                    flag = True
                                    break

                    if (flag == False):
                        dict = {'start': frame, 'row': row, 'col': col}
                        dict_vec.append(dict)

    N = len(dict_vec)
    trajectories_arr = np.zeros((N, MaxTraceLength))
    coordinates = np.zeros((N, 2))
    for i in range(N):
        start = dict_vec[i]['start']
        stop = dict_vec[i]['start']+MaxTraceLength
        if(stop > total_frames):
            stop = total_frames
            start = total_frames - MaxTraceLength
        trajectories_arr[i, :] = raw_data[start:stop, dict_vec[i]['row'], dict_vec[i]['col']]
        coordinates[i, 0] = dict_vec[i]['row']
        coordinates[i, 1] = dict_vec[i]['col']

    print("-I- found", N, "time traces")
    return trajectories_arr, coordinates

def feature_extraction(trajectories, threshold, numOfBins):
    """
        Extracting n_blinks features according to the given trajectories
        :param trajectories: List of trajectories
        :param threshold: Float for minimal threshold for peak detection
        :param numOfBins: Int for the range of the n_blinks histogram
        :param tau: Int for the tau_c parameter
        :return: Tensor [1, numOfBins] for the n_blinks histogram of the experiment.
    """
    L = len(trajectories)
    features = np.zeros((L, numOfBins))
    n_blinks_per_cluster = []
    for i in range(L):
        numOfTrajectories, TrajectoryLength = trajectories[i].shape
        for trajectory in range(numOfTrajectories):
            n_blinks = 0
            frame = 0
            while(frame < TrajectoryLength):
                if(trajectories[i][trajectory, frame] >= threshold):
                    n_blinks += 1
                    start_time = frame + 10
                    while (trajectories[i][trajectory, frame] >= threshold and frame < start_time):
                        frame += 1
                        if (frame > TrajectoryLength - 1):
                            break
                else:
                    if(trajectories[i][trajectory, frame] < threshold/2):
                        while (trajectories[i][trajectory, frame] < threshold):
                            frame += 1
                            if (frame > TrajectoryLength - 1):
                                break
                    else:
                        frame += 1

            if (n_blinks > 0 and n_blinks < numOfBins):
                features[i, n_blinks - 1] += 1
                n_blinks_per_cluster.append(n_blinks)
            else:
                n_blinks_per_cluster.append(0)

    np.save('X_test', features)
    return features,  np.expand_dims(np.array(n_blinks_per_cluster), axis=1)

def LoadFinalDataSet():
    """
        Loads an existing data set after feature extraction
        :return: data set [# of experiments, numOfBins].
    """
    data_set = np.load('X_test.npy')
    return data_set

def CreateSimulatedDataSet_special():
    path = 'D:\Project\data'
    files = [r'CTLA4/mEos3.2.tif', r'CTLA4/mEos3.2_X2.tif']
    data = np.zeros((24000, 251, 257))

    for j in range(2):
        tiff = tc.opentiff(os.path.join(path, files[j]))
        for i in range(12000):
            data[int(12000*j + i), :, :] = tiff.find_and_read(i)

    return data

def CreateSimulatedDataSet(data_set_size, num_of_molecules, bleach_proba, numOfBins):
    """
        Creates many simulations of features based on ground truth
        :param data_set_size: Int for the number of simulations to perform
        :param num_of_molecules: Int for the number of molecules to simulate per simulations
        :param bleac_proba: Float for the bleaching probability of single cluster
        :param numOfBins: Int for the range of the n_blinks histogram
        :return: data_set [data_set_size, numOfBins] for the entire simulated data set.
    """
    X = np.zeros([data_set_size, numOfBins])
    y = np.zeros(data_set_size)

    for i in range(data_set_size):
        if(np.random.rand() > 0.4): # Creating more monomers only experiments
            y[i] = 0 # Monomers only
        else:
            y[i] = np.random.rand() # Random dimers percentage

        MonoList = []
        for j in range(int(num_of_molecules - num_of_molecules * y[i])): # MONOMERS
            MonoCnt = 1 # Always start with one blink
            while(np.random.rand() > bleach_proba): # add blinks until bleaches
                MonoCnt += 1
            MonoList.append(MonoCnt)
        DimersList = []
        for j in range(int(num_of_molecules - num_of_molecules * y[i]), num_of_molecules): # DIMERS
            DimersCnt = 2 # Always start with two blinks
            while(np.random.rand() > bleach_proba): # add blinks until bleaches
                DimersCnt += 1
            while (np.random.rand() > bleach_proba): # dimers bleach twice
                DimersCnt += 1
            DimersList.append(DimersCnt)

        Nblinks_mono = np.histogram(np.array(MonoList) - 1, bins=numOfBins, range=(0, numOfBins))[0]
        Nblinks_dimers = np.histogram(np.array(DimersList) - 1, bins=numOfBins, range=(0, numOfBins))[0]

        X[i, :] = Nblinks_mono + Nblinks_dimers

    np.save('X', X)
    np.save('y', y)
    return X, y

def LoadSimulatedDataSet_special():
    """
        Loads an existing trajectories list
        :return: List of trajectories.
    """
    X = np.load('X_special.npy')
    y = np.load('y_special.npy')
    return X, y

def LoadSimulatedDataSet():
    """
        Loads an existing trajectories list
        :return: List of trajectories.
    """
    X = np.load('X.npy')
    y = np.load('y.npy')
    return X, y

def Filter_beads(data):
    mean_intensity = np.mean(data)
    std_intensity = np.std(data)
    bids_loc = np.where(np.mean(data, axis=0) > mean_intensity * 1.1)
    data[:, bids_loc[0], bids_loc[1]] = np.random.normal(mean_intensity, std_intensity, [data.shape[0], len(bids_loc[0])])
    return data

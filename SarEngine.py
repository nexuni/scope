import csv
import sys
import yaml
from typing import Dict, Any
import numpy as np
import time

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as patches

ncolors = 256
color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)

class SarEngine:
    """
    This is the SAR Engine for UHF RFID localization.
    """
    def __init__(self, c=299792458):
        """
        :param c: light speed in (m/s)
        """
        self.c = c

        self.xy_gridsize = 0.01
        self.z_gridsize = 0.2
        self._update_tag_bound((0, 6), (1.2, 1.8), (0.45, 0.65))

        # Simulation read history
        self.array_read_history = {}
        self.antenna_array_dict = {}
        self.localization_dict = {}
        self.total_pxyz = {}

        self.conf_threshold =  np.array([0.3, 0.3, 0.601])
        self.min_conf = np.array([0.05, 0.05, 0.05]) #(3, ): minimum X,Y,Z confidence intervals
        self.near_edge_heuristic = 1 #scalar: how much to grow region by if location is near edge
        self.edge_frac = 0.2 #scalar: how much of region is considered "near edge"
        self.db = 0.75 #scalar: db off of max for confidence interval

    def _compute_confidence_intervals(self, p_xyz, peak_ind):
        ''' Compute 3D confidence interval
        1) Confidence is based on the bounding box of the region that is some dB below the peak
        2) Confidence intervals cannot be less than the min confidence
        3) Confidence intervals are returned in meters
        4) If the guess is near the edge of the search region, multiply the confidence by the given parameter to increase the region
        '''
        # Compute 1dB mask
        db_frac = 10**(self.db/20)
        # peak_ind = (peak_ind[1], peak_ind[0], peak_ind[2])
        peak_val = p_xyz[peak_ind]
        mask = np.where(p_xyz < (peak_val/db_frac), 0, 1)

        # Find bounding box around 1dB region
        # Code from: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        x = np.any(mask, axis=(0, 2))
        y = np.any(mask, axis=(1, 2))
        z = np.any(mask, axis=(0, 1))

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        # Select confidence as maximum of (peak - left) and (right - peak)
        # NOTE: Peak_ind = (y,x,z)
        x_conf = max(peak_ind[1] - xmin, xmax - peak_ind[1])*self.xy_gridsize
        y_conf = max(peak_ind[0] - ymin, ymax - peak_ind[0])*self.xy_gridsize
        z_conf = max(peak_ind[2] - zmin, zmax - peak_ind[2])*self.z_gridsize

        # print("X confidence is ", x_conf, " , X min is ", xmin, " , X max is ", xmax, " , Opt #1: ", peak_ind[0] - xmin, " , Opt #2: ", xmax - peak_ind[0],  " , peak: ", peak_ind[0] )
        # print("Y confidence is ", y_conf, " , Y min is ", ymin, " , Y max is ", ymax, " , Opt #1: ", peak_ind[1] - ymin, " , Opt #2: ", ymax - peak_ind[1], " , peak: ", peak_ind[1] )
        # print("Z confidence is ", z_conf, " , Z min is ", zmin, " , Z max is ", zmax, " , Opt #1: ", peak_ind[2] - zmin, " , Opt #2: ", zmax - peak_ind[2],  " , peak: ", peak_ind[2] )
        # print("Widths are ", widths[0], widths[1], widths[2])
        # print("-----------------")


        # Ensure confidence is above minimum
        # if x_conf < self.min_conf[0]: print('X at min conf')
        # if y_conf < self.min_conf[1]: print('Y at min conf')
        # if z_conf < self.min_conf[2]: print('Z at min conf')
        x_conf = max(x_conf, self.min_conf[0])
        y_conf = max(y_conf, self.min_conf[1])
        z_conf = max(z_conf, self.min_conf[2])

        # Grow region if location is near edge of search box
        # NOTE: Peak_ind = (y,x,z)
        # print(f'Peak at ({peak_ind[0]},{peak_ind[1]},{peak_ind[2]})')
        if self._is_loc_near_edge(self.grid_range[0], peak_ind[1], self.edge_frac): x_conf *= self.near_edge_heuristic
        if self._is_loc_near_edge(self.grid_range[1], peak_ind[0], self.edge_frac): y_conf *= self.near_edge_heuristic
        if self._is_loc_near_edge(self.grid_range[2], peak_ind[2], self.edge_frac): z_conf *= self.near_edge_heuristic

        # print(f'Final conf interval: ({x_conf},{y_conf},{z_conf})')
        return x_conf, y_conf, z_conf

    def _is_loc_near_edge(self, grid_size, peak_ind, frac):
        ''' Check if 1 dimension of location is near edge of grid
        A location is "near the edge" if it is within some fraction of the grid points near the edge (e.g. top or bottom 1/3 of grid)
        '''
        return peak_ind < grid_size*frac or peak_ind > grid_size - grid_size*frac

    def _update_tag_bound(self, xbound, ybound, zbound):
        self.tag_bound = [xbound, ybound, zbound]

        # Confidence interval
        xgridrange = int((self.tag_bound[0][1] - self.tag_bound[0][0]) / self.xy_gridsize)
        ygridrange = int((self.tag_bound[1][1] - self.tag_bound[1][0]) / self.xy_gridsize)
        zgridrange = int((self.tag_bound[2][1] - self.tag_bound[2][0]) / self.z_gridsize)
        self.grid_range = np.array([xgridrange, ygridrange, zgridrange])

        self.state_x_locs = np.arange(self.tag_bound[0][0], self.tag_bound[0][1]+self.xy_gridsize, self.xy_gridsize)
        self.state_y_locs = np.arange(self.tag_bound[1][0], self.tag_bound[1][1]+self.xy_gridsize, self.xy_gridsize)
        self.state_z_locs = np.arange(self.tag_bound[2][0], self.tag_bound[2][1]+self.z_gridsize, self.z_gridsize)

    def _obs_to_array(self, obs):
        """
        :param obs: A list of (rssi, phase, freq)
        :return: A numpy array of (rssi, phase, freq)
        """
        rssi_list = []
        phase_list = []
        freq_list = []
        for i in range(len(obs)):
            rssi_list.append(obs[i][0])
            phase_list.append(obs[i][1])
            freq_list.append(obs[i][2])

        return np.array(rssi_list), np.array(phase_list), np.array(freq_list)

    def compute_near_field(self, traj, amp_arr, unwrapped_phase_arr, freq_arr, mask=None):
        x_locs = np.arange(self.tag_bound[0][0], self.tag_bound[0][1]+self.xy_gridsize, self.xy_gridsize)
        y_locs = np.arange(self.tag_bound[1][0], self.tag_bound[1][1]+self.xy_gridsize, self.xy_gridsize)
        z_locs = np.arange(self.tag_bound[2][0], self.tag_bound[2][1]+self.z_gridsize, self.z_gridsize)

        xv, yv, zv = np.meshgrid(x_locs, y_locs, z_locs)
        d = np.zeros((len(y_locs), len(x_locs), len(z_locs), len(traj), 1)) # 1 means 1 frequency for now
        for i in range(len(traj)):
            tmp = np.sqrt(np.square(traj[i][0] - xv) + np.square(traj[i][1] - yv) + np.square(traj[i][2] - zv)) \
                  + np.sqrt(np.square(traj[i][0] - xv) + np.square(traj[i][1] - yv) + np.square(traj[i][2] - zv))
            tmp = np.expand_dims(tmp, axis=3)
            d[:, :, :, i, :] = np.tile(tmp, (1, 1, 1, 1)) # np.tile(tmp, (1, 1, 1, channels.shape[1]))

        N = len(traj)
        K = 1 # channels.shape[1]
        # exp_ = np.divide(d, np.array([self.c/self.freq]))
        # print("exp shape: ", exp_.shape)
        freq_divisor = np.array([self.c/f for f in freq_arr])
        freq_divisor = freq_divisor.reshape((1,1,1,N,K))
        exp_ = np.divide(d, freq_divisor)
        channels = np.e**(-1j*unwrapped_phase_arr) #amp_arr
        channels = channels.reshape((unwrapped_phase_arr.shape[0],1))

        # if mask is not None:
        #     mask = mask.reshape((len(y_locs), len(x_locs), len(z_locs), 1, 1))
        #     p_xyz = np.abs(1/(K*N)*np.sum(np.sum(np.multiply(np.multiply(channels, np.e**(1j*2*np.pi*exp_)), mask), axis=4), axis=3))
        # else:
        p_xyz = np.abs(1/(K*N)*np.sum(np.sum(np.multiply(channels, np.e**(1j*2*np.pi*exp_)), axis=4), axis=3))

        return x_locs, y_locs, z_locs, p_xyz

    def process_sar(self, tag, actual=False, mask=None):
        """
        Only process sar for that specific tag. Because the latest
        read is only filtered against that specific tag.
        :param tag:
        :param actual:
        :param mask:
        :param coarse_to_fine:
        :return:
        """
        # Guard no new read
        if tag not in self.array_read_history.keys():
            return

        tag_read_data = self.array_read_history[tag][-1]
        # Get the observation array from obs
        rssi_arr, phase_arr, freq_arr = self._obs_to_array(tag_read_data["obs"])

        # RSSI is in db, convert to amplitude
        amplitude_arr = 10**(rssi_arr/20)

        # Based on the raw data, raw data is already within 180, so PHASE and UNWRAPPED PHASE should be the same?
        if actual:
            rad_phase_arr = np.pi - np.deg2rad(phase_arr)
        else:
            rad_phase_arr = phase_arr

        # Pi Unwrapping / ThingMagic 180 wrapping
        rad_phase_unwrapped_arr = np.unwrap(rad_phase_arr, period=np.pi)

        xlocs, ylocs, zlocs, pxyz = self.compute_near_field(tag_read_data["traj"], amplitude_arr, rad_phase_unwrapped_arr, freq_arr, mask=mask)

        if tag not in self.total_pxyz:
            self.total_pxyz[tag] = pxyz
        else:
            if pxyz.shape[1] > self.total_pxyz[tag].shape[1]:
                pxyz = pxyz[:, :self.total_pxyz[tag].shape[1], :]
            else:
                self.total_pxyz[tag] = self.total_pxyz[tag][:, :pxyz.shape[1], :]
            self.total_pxyz[tag] += pxyz

    # @profile
    def localize_tag(self, tag):
        # for tag in self.active_tags:
        # Guard no new read
        if tag not in self.array_read_history.keys():
            return

        # Get the last measurement
        tag_loc = [0,0,0]#self.array_read_history[tag][-1]["gt"]

        p_xyz = self.total_pxyz[tag]

        arg_for_error_compute = [tag_loc, self.state_x_locs.tolist(), self.state_y_locs.tolist(), self.state_z_locs.tolist(), p_xyz]
        err, peak, peak_ind = self.compute_near_field_error(*arg_for_error_compute)
        conf_interval = self._compute_confidence_intervals(p_xyz, peak_ind)

        try:
            self.localization_dict[tag].append((peak, err, 0, conf_interval))
        except:
            self.localization_dict[tag] = [(peak, err, 0, conf_interval)]

    def update_scope_data(self, data, tag_info, xoffset=[-1, 1], yoffset=[0.4,1.2], zoffset=[0.6,0.65]):
        ret_data = {}
        for tag in data.keys():
            if len(data[tag]["traj"]) == 0:
                continue
            total_scan_traj = []
            xx_list = []
            yy_list = []
            for scan_i in range(len(data[tag]["obs"])):
                for i in range(len(data[tag]["traj"][scan_i])):
                    xx_list.append(data[tag]["traj"][scan_i][i][0])
                    yy_list.append(data[tag]["traj"][scan_i][i][1])

            mid_x = np.mean(xx_list)
            mid_y = np.mean(yy_list)
            self._update_tag_bound((mid_x + xoffset[0], mid_x + xoffset[1]), (mid_y + yoffset[0], mid_y + yoffset[1]), (zoffset[0], zoffset[1]))

            for scan_i in range(len(data[tag]["obs"])):
                # calculate a list of cumulative distance traveled from the first point of the trajectory
                # to the current point
                kd = np.cumsum(np.linalg.norm(np.diff(data[tag]["traj"][scan_i], axis=0), axis=1))
                tag_read_dict = {"obs": data[tag]["obs"][scan_i],
                                 "traj": data[tag]["traj"][scan_i],
                                 "kD": np.concatenate((np.array([0]), kd))}

                total_scan_traj.append(tag_read_dict["traj"])
                try:
                    self.array_read_history[tag].append(tag_read_dict)
                except:
                    self.array_read_history[tag] = [tag_read_dict]

                # Update antenna array and localization one scan at a time
                self.process_sar(tag, actual=True)
                self.localize_tag(tag)

            if tag in self.total_pxyz:
                # Get top 50 peaks
                flat_indices = np.argsort(self.total_pxyz[tag].ravel())[-500:][::-1]
                peak_indices = np.unravel_index(flat_indices, self.total_pxyz[tag].shape)

                x_loc_array = peak_indices[1] * self.xy_gridsize + mid_x + xoffset[0]
                y_loc_array = mid_y + yoffset[1] - peak_indices[0] * self.xy_gridsize
                z_loc_array = peak_indices[2] * self.z_gridsize + zoffset[0]

                ret_data[tag] = {"traj": total_scan_traj, "state_x_locs": self.state_x_locs.tolist(),
                             "state_y_locs": self.state_y_locs.tolist(), "state_z_locs": self.state_z_locs.tolist(),
                             "total_pxyz": self.total_pxyz[tag], "top_50_peaks": (x_loc_array, y_loc_array, z_loc_array)}

        return ret_data

    def compute_near_field_error(self, tag_loc, x_locs, y_locs, z_locs, p_xyz):
        peak_ind = np.unravel_index(np.argmax(p_xyz, axis=None), p_xyz.shape)
        peak = np.array([x_locs[peak_ind[1]], y_locs[peak_ind[0]], z_locs[peak_ind[2]]])
        return np.linalg.norm(peak - np.array(tag_loc)), peak, peak_ind

    def plot_near_field_sigcomm(self, x_locs, y_locs, z_locs, z_slice, p_xyz, tag_loc, rx_antennas, add_border=False, peak=None, peak_ind=None, x_bounds=None, y_bounds=None, plt_module=None):
        if plt_module is not None:
            plt_ = plt_module
        else:
            plt_ = plt

        if peak is None or peak_ind is None:
            peak_ind = np.unravel_index(np.argmax(p_xyz, axis=None), p_xyz.shape)
            peak = (x_locs[peak_ind[1]], y_locs[peak_ind[0]], z_locs[peak_ind[2]])
            print('Peak: ', peak)

        peak_val = np.amax(p_xyz, axis=None)
        selected = np.where(p_xyz[:,:,z_slice] < peak_val/1.2, p_xyz[:,:,z_slice], 0)
        selected = np.where(peak_val/1.25 < selected , 1, 0)

        # selected_small = np.where(p_xy <= peak_val/1, p_xy, 0)
        # selected_small = np.where(peak_val/1.012 < selected_small , 1, 0)

        normalize = Normalize(vmin=np.min(p_xyz, axis=None), vmax=np.max(p_xyz, axis=None))
        plt_.pcolormesh(x_locs, y_locs, p_xyz[:,:,z_slice], norm=normalize)

        if len(rx_antennas) > 1:
            for i in range(len(rx_antennas)):
                plt_.scatter(rx_antennas[i, 0], rx_antennas[i,1], label=i)
            plt_.scatter(rx_antennas[:, 0], rx_antennas[:,1], label='RX Antennas', color='pink')

        # if peak_ind[2] == z_slice: plt_.scatter(peak[0], peak[1], color='g', label='guess')
        # plt_.scatter(tag_loc[0], tag_loc[1], color='r', label='tag')

        # plt_.colorbar()
        # plt_.set_title(f'Simulated data, z={z_locs[z_slice]}')
        if x_bounds != None and y_bounds != None:
            plt_.xlim(x_bounds)
            plt_.ylim(y_bounds)
        # plt_.show()

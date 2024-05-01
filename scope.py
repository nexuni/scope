import pandas as pd
import sys
import difflib
import json
import numpy as np
from scipy import constants
from SarEngine import SarEngine
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Scope:
    """
    This is the main class of scope. It takes in three dataframes and
    process the matching between the plate number and its corresponding
    lot number.

    - Input:
        * scope_pos_df: The absolute position of the scooter location
        at each timestamps. The entry includes:
            ** x, y, z, Timestamps
        * plate_read_df: The license plate recognition results and the
        relative position of license plate. The entry includes:
            ** plate_num, x, y, z, Timestamps
        * rf_df: The RF measurement results from ThingMagic. The entry
        includes:
            ** EPC, phase, amplitude, frequency, timestamps
        * ground_truth_info: A dictionary to match RFID EPC ID to the correct
        license plate number.
    """
    def __init__(self, scope_pos_df, plate_read_df, rf_read_df, ground_truth_info):
        self.scope_pos_df = scope_pos_df
        self.plate_read_df = plate_read_df
        self.rf_read_df = rf_read_df
        self.ground_truth_info = ground_truth_info

        self.engine = SarEngine()
        self.preprocess_dataframe()

        # Define window and step sizes in seconds
        # Window size means how many measurements we are
        # going to aggregate to get the SAR computation. (And visualize)
        self.window_size = 10 # seconds
        self.step_size = 2 # seconds

    def _interpolate_positions(self, time_a, time_b, positions_b):
        """
        Interpolate x, y positions at measurement times.

        :param time_a: List of datetime objects for measurements.
        :param time_b: List of datetime objects for position feedbacks.
        :param positions_b: List of (x, y) tuples for each feedback time in time_b.
        :return: List of interpolated (x, y) positions at each measurement time in time_a.
        """
        # Convert datetime objects to timestamps
        time_a_stamps = time_a
        time_b_stamps = time_b

        # Separate x and y positions
        # x_positions = zip(*positions_b)

        # Interpolate x and y positions
        x_interpolated = np.interp(time_a_stamps, time_b_stamps, positions_b)
        # y_interpolated = np.interp(time_a_stamps, time_b_stamps, y_positions)

        # Combine x and y positions into tuples
        interpolated_positions = list(x_interpolated)

        return interpolated_positions

    def _filter_recognized_plate_no(self, grouped_plate_pos):
        """
        Filter recognized plate number. For now we are filtering
        based on ground truth plate no. More sophisticated filtering
        can be done on real deployments to get the max agreement based
        on multiple observations.
        :param grouped_plate_pos:
        :return:
        """
        def find_most_similar(target, strings):
            """
            Finds the string most similar to the target string in a list of strings.

            :param target: String to which the comparison is made.
            :param strings: List of strings to compare against the target.
            :return: The most similar string to the target.
            """
            # Initialize the most similar string and the highest similarity score
            most_similar = None
            highest_similarity = 0.0

            # Iterate over each string in the list
            for string in strings:
                # Calculate the similarity ratio
                similarity = difflib.SequenceMatcher(None, target, string).ratio()

                # Check if the calculated similarity is higher than the highest recorded one
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar = string

            return most_similar

        filtered_dict = {}
        for group in grouped_plate_pos:
            plate_no = find_most_similar(group, list(self.ground_truth_info.values()))
            try:
                filtered_dict[plate_no] += grouped_plate_pos[group]
            except:
                filtered_dict[plate_no] = grouped_plate_pos[group]

        return filtered_dict

    def preprocess_dataframe(self):
        """
        The main preprocess task performed here are:
        1. Set the indices of the dataframes to timestamp of same unit
        2. interpolate positions from scope pos to rf df to provide relative
        location for SAR computation
        3. Compute the absolute position of recognized license plate
        :return:
        """
        # Preprocess (1)
        self.scope_pos_df['Timestamp'] = self.scope_pos_df['Timestamp'] / 1e9
        self.scope_pos_df = self.scope_pos_df.set_index('Timestamp')
        self.plate_read_df['Timestamp'] = self.plate_read_df['Timestamp'] / 1e9
        self.plate_read_df = self.plate_read_df.set_index('Timestamp')
        self.rf_read_df = self.rf_read_df.set_index('Timestamp')

        # Preprocess (2)
        interpolocations_x = self._interpolate_positions(self.rf_read_df.index, self.scope_pos_df.index, self.scope_pos_df['x'].values)
        interpolocations_y = self._interpolate_positions(self.rf_read_df.index, self.scope_pos_df.index, self.scope_pos_df['y'].values)
        self.rf_read_df['x'] = interpolocations_x
        self.rf_read_df['y'] = interpolocations_y

        # Preprocess (3)
        abs_x_list, abs_y_list = [], []
        for index, row in self.plate_read_df.iterrows():
            closest_index_value = self.scope_pos_df.index[np.argmin(np.abs(self.scope_pos_df.index - index))]
            abs_x_list.append(self.scope_pos_df.loc[closest_index_value]['x'] + row['x'])
            abs_y_list.append(self.scope_pos_df.loc[closest_index_value]['y'] + row['y'])

        self.plate_read_df['abs_x'] = abs_x_list
        self.plate_read_df['abs_y'] = abs_y_list

    def main(self, plot=False):
        # Plot SAR
        data_filtered_all = self.filter_valid_arrays_and_tags(self.rf_read_df, self.ground_truth_info, interval=0.125)
        rf_data = self.engine.update_scope_data(data_filtered_all, self.ground_truth_info, xoffset=[-1, 1], yoffset=[0.5,1.3], zoffset=[0.5,0.55])
        # self.plot_scope_lot(len(rf_data.keys()), rf_data, plt_func=self.engine.plot_near_field_sigcomm)

        # Assuming all dataframes are indexed by datetime and have the same overall time range
        start_time = self.scope_pos_df.index.min()
        end_time = self.scope_pos_df.index.max()

        # Adjust end_time based on the window size to avoid an incomplete window at the end
        end_time -= self.window_size

        # Loop through time range with given step size
        current_time = start_time
        while current_time <= end_time:
            window_end = current_time + self.window_size

            # Filter each dataframe to the current window
            scope_pos_window = self.scope_pos_df.loc[current_time:window_end]
            plate_read_window = self.plate_read_df.loc[current_time:window_end]

            grouped_plate_pos = plate_read_window.groupby('plate_number').apply(lambda df: df[['abs_x', 'abs_y']].values.tolist()).to_dict()
            filtered_dict = self._filter_recognized_plate_no(grouped_plate_pos)

            self.plot_scope(scope_pos_window, filtered_dict, rf_data)

            # Increment current_time by the step size
            current_time += self.step_size

    ##########################################################
    ######       RFID SAR Computation Related           ######
    ##########################################################
    def filter_valid_arrays_and_tags(self, data, tag_info, interval=0.125):
        """
        correspond the target tag with the array and filter out valid arrays based
        on interval and min measurement check.
        :return:
        """
        output_data = {}
        for x, y, epc_, rssi, phase, freq in zip(data['x'], data['y'], data['EPC'], data['RSSI'], data['Phase'], data['Frequency']):
            # EPC string handling
            epc = epc_.replace("'", "")
            epc = epc[1:]
            if epc not in tag_info:
                continue

            try:
                output_data[epc]["obs"].append((rssi, phase, freq*1000.0))
                output_data[epc]["traj"].append((x, y, 0))
            except:
                output_data[epc] = {
                    "obs": [(rssi, phase, freq*1000.0)],
                    "traj": [(x, y, 0)]
                }

        def filter_data(epc, data, lambda_ratio_interval=interval, min_measurement=10, min_aperture=0.15): #15, 0.5
            fdata = {"obs": [], "traj": []}
            coordinates = data[epc]["traj"]
            max_interval_split = lambda_ratio_interval * constants.c / (data[epc]["obs"][0][2])
            # print("Wavelength: ", constants.c / (data[epc]["obs"][0][2]))

            current_cluster = [coordinates[0]]
            current_indices = [0]

            for idx, point in enumerate(coordinates[1:], start=1):
                # Calculate distance from current point to the last point in the current cluster
                dist = np.linalg.norm(np.array(point) - np.array(current_cluster[-1]))

                # If distance is greater than max_interval_split, a new cluster begins
                if dist > max_interval_split:
                    # If current cluster size reaches min_cluster_size, we keep it
                    if len(current_cluster) >= min_measurement and np.linalg.norm(np.array(current_cluster[-1]) - np.array(current_cluster[0])) >= min_aperture:
                        fdata["obs"].append([data[epc]["obs"][i] for i in current_indices])
                        fdata["traj"].append([data[epc]["traj"][i] for i in current_indices])

                    current_cluster = [point]
                    current_indices = [idx]
                else:
                    current_cluster.append(point)
                    current_indices.append(idx)

            # Append the last cluster if it's not empty and has at least min_cluster_size points
            if current_cluster and len(current_cluster) >= min_measurement \
                    and np.linalg.norm(np.array(current_cluster[-1]) - np.array(current_cluster[0])) >= min_aperture:
                fdata["obs"].append([data[epc]["obs"][i] for i in current_indices])
                fdata["traj"].append([data[epc]["traj"][i] for i in current_indices])

            return fdata

        filtered_data = {}
        for epc_ in output_data.keys():
            filtered_data[epc_] = filter_data(epc_, output_data)

        return filtered_data

    ##########################################################
    #####              Plotting Related                 ######
    ##########################################################
    def _find_map_bound(self, padding=0.5):
        """
        This is the function to find the bound of the
        scope parking enforcement map based on the scope
        position dataframe.
        - Input:
            * padding: the padding of the map bound from
            the min and max of the scope pos values
        :return: xbound, ybound
        """
        max_x = max(df.max() for df in [self.scope_pos_df["x"], self.plate_read_df["abs_x"]])
        min_x = min(df.min() for df in [self.scope_pos_df["x"], self.plate_read_df["abs_x"]])

        max_y = max(df.max() for df in [self.scope_pos_df["y"], self.plate_read_df["abs_y"]])
        min_y = min(df.min() for df in [self.scope_pos_df["y"], self.plate_read_df["abs_y"]])

        return (min_x-padding, max_x + padding), (min_y-padding, max_y + padding)

    def plot_scope(self, scope_pos_data_window, grouped_plate_data_window, rf_data, scope_length=1.0, scope_width=0.6):
        """
        This is the function to find the orientation
        of the scooter based on the movement direction
        of the scooter in one window.
        - Input:
            * scope_pos_data_window: the scope position information
            in one window.
            * scope_length: scooter length
            * scope_width: scooter width
        """
        x = scope_pos_data_window["x"]
        y = scope_pos_data_window["y"]

        # Perform linear regression to find the best fit line
        slope, intercept = np.polyfit(x, y, 1)

        # Plot data points and the regression line
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o', label='Data points')

        # Calculate the midpoint of the line for placing the rectangle
        mid_x = np.mean(x)
        mid_y = np.mean(y)

        # Angle of rotation in radians
        theta = np.arctan(slope)
        # Angle of rotation in degrees
        angle = theta * (180 / np.pi)

        # Create a rectangle. Apply rotation matrix.
        rect = patches.Rectangle((mid_x - scope_length/2* np.cos(theta) + scope_width/2* np.sin(theta), mid_y - scope_length/2* np.sin(theta) - scope_width/2* np.cos(theta)), scope_length, scope_width, angle=angle,
                                 linewidth=1, edgecolor='blue', facecolor='none')

        # Add the rectangle to the plot
        plt.gca().add_patch(rect)

        xbound, ybound = self._find_map_bound()
        for plate_no in grouped_plate_data_window:
            meas_data = np.array(grouped_plate_data_window[plate_no])
            plt.scatter(meas_data[:,0], meas_data[:,1], color='pink', marker='o', s=100, label='Scope Position', alpha=0.4)
            plt.text(np.mean(meas_data[:,0]), np.mean(meas_data[:,1])+0.3, plate_no, fontsize=12, color='red', ha='center', va='center')

        for tag in rf_data.keys():
            peaks_locations = rf_data[tag]["top_50_peaks"]
            plt.scatter(peaks_locations[0], peaks_locations[1], color='green', marker='o', s=100, label='Scope Position', alpha=0.4)
            plt.text(np.mean(peaks_locations[0]), np.mean(peaks_locations[1])+0.5, tag[-6:], fontsize=12, color='green', ha='center', va='center')

        # Setting plot labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Best Fit Line and Aligned Rectangle')
        plt.xlim(*xbound)
        plt.ylim(*ybound)
        plt.grid(True)
        plt.show()

    def plot_scope_lot(self, num_tags, rf_data, plt_func=None):
        fig, axs = plt.subplots(round(num_tags/2 + 0.5), 2, figsize=(15, 8))
        for i, tag in enumerate(rf_data.keys()):
            axs[i//2][i%2].set_title(f"SAR RF Probability Heatmap for Lot: {self.ground_truth_info[tag]}")
            xbound, ybound = self._find_map_bound()
            axs[i//2][i%2].set_xlim(*xbound)
            axs[i//2][i%2].set_ylim(*ybound)
            traj = []
            for j in range(len(rf_data[tag]["traj"])):
                traj.extend(rf_data[tag]["traj"][j])

            traj_array = np.array(traj)
            peak_ind = np.unravel_index(np.argmax(rf_data[tag]["total_pxyz"], axis=None), rf_data[tag]["total_pxyz"].shape)
            plt_func(rf_data[tag]["state_x_locs"], rf_data[tag]["state_y_locs"], rf_data[tag]["state_z_locs"], peak_ind[2], rf_data[tag]["total_pxyz"], [0,0,0], traj_array, plt_module=axs[i//2][i%2])

        plt.subplots_adjust(hspace=0.8)
        plt.show()

if __name__ == "__main__":
    with open('./tags/lot_tag_info_0428.json') as f:
        ground_truth_info = json.load(f)

    # This is from exp_data_0428
    camera_epochtime = "1714097347" #sys.argv[1]
    rf_epochtime = "1714097331" #sys.argv[2]

    # This is from exp_data_0327
    # camera_epochtime = "1711535177" #sys.argv[1]
    # rf_epochtime = "1711535210" #sys.argv[2]

    # Prepare the file path
    scope_abs_pos = f'./exp_data_0428/absolute_move_{camera_epochtime}.csv'
    scope_plate_read = f'./exp_data_0428/plate_reads_{camera_epochtime}.csv'
    scope_rf_read = f'./exp_data_0428/rfid_reads_{rf_epochtime}.csv'

    # Reading the CSV file
    scope_pos_df = pd.read_csv(scope_abs_pos)
    plate_read_df = pd.read_csv(scope_plate_read)
    rf_read_df = pd.read_csv(scope_rf_read)

    scope = Scope(scope_pos_df, plate_read_df, rf_read_df, ground_truth_info)
    scope.main(plot=True)

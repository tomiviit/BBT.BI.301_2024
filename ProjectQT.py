from statistics import variance

import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import glob
import os
import neurokit2 as nk
from scipy.signal import sosfilt, iirnotch, butter, tf2sos
from scipy import stats
import seaborn as sns

""" This piece of code downloads the ECG data, calculates QT intervals and performs the 
anova tests for different genotype groups (AA, AB, BB). The ECG data analysis, i.e., calculating the
QT intervals is ready-made for you. Finding the associations between different groups is done by you. 
Hint for these are in the code! Read all the comments carefully!!"""

# MODIFY YOUR PATH
path_new = 'C:/Users/Tomi Viitanen/binf_python/Project/new/'
path_old = 'C:/Users/Tomi Viitanen/binf_python/Project/ecg/'

# for creating a dict of the filenames and corresponding ids so I can
# use the old datasets and get actual results
def extract_ids(directory):
    hea_dict = {}

    # Loop through file numbers 1 to 129
    for i in range(1, 130):
        filename = os.path.join(directory, f'{i}.hea')

        # Open and read the first line of the file
        with open(filename, 'r') as file:
            first_line = file.readline().strip()

            # Extract the first word from the line
            first_word = first_line.split()[0]

            # Store it in the dictionary with the file number as the key
            hea_dict[i] = first_word

    return hea_dict

# This function loads the ECG data files: .dat, and .hea -files. .dat files consists of tne ECG voltage data
# and .hea files consist of the header information (e.g., sampling frequency, leads etc.)
def loadEcg(path, id_dict):
    ecg_dict, field_dict, fs_dict = {}, {}, {}

    # Read the signal and metadata for each file. The read file consist of field names (contain, e.g.,
    # the sampling frequency and the lead names), and the actual ecg signal data.
    for i in range(1,130):
        base_name = id_dict[i]
        ecg, fields = wfdb.rdsamp(os.path.join(path, base_name))

        patient_key = f'Patient{i}'
        ecg_dict[patient_key] = ecg
        field_dict[patient_key] = fields
        fs_dict[patient_key] = fields['fs']
    return ecg_dict, field_dict, fs_dict


# The bandpass filter (0.7hz-150hz) isolates relevant frequensies from outside noise.
# 0.7hz lowcut removes baseline wondering. 60hz notch filter is there to remove powerline
# interference. Using SOS (second order sections) ensures no phase distortion to preserve the
# shape of the ECG signal.
def filterEcg(signal, fs, filter_order=2, low=0.7, high=150, notch_freq=60, notch_q=30):

    nyquist = fs / 2
    low_cutoff = low / nyquist
    high_cutoff = high / nyquist

    sos_bandpass = butter(filter_order, [low_cutoff, high_cutoff], btype='band', output='sos')

    leadII = signal[:, 1]

    filtered_signal = sosfilt(sos_bandpass, leadII)

    # Filter for 60Hz powerline interference
    notch_b, notch_a = iirnotch(notch_freq / nyquist, notch_q)

    # Convert notch filter to second-order sections (SOS)
    sos_notch = tf2sos(notch_b, notch_a)

    filtered_signal = sosfilt(sos_notch, filtered_signal)

    return filtered_signal


# Function to calculate QT intervals.
    # Process the ECG signal using neurokit2 (detects R-peaks, Q, and T points)
    # ecg_process has many steps:
    # 1) cleaning the ecg with ecg_clean()
    # 2) peak detection with ecg_peaks()
    # 3) HR calculus with signal_rate()
    # 4) signal quality assessment with ecg_quality()
    # 5) QRS elimination with ecg_delineate() and
    # 6) cardiac phase determination with ecg_phase().
def calculateQtIntervals(key, filtered_signal, fs, plot):
    ecg_analysis, _ = nk.ecg_process(filtered_signal, sampling_rate=fs)
    q_points = ecg_analysis['ECG_Q_Peaks'] # This is default output of the ecg_process.
    t_points = ecg_analysis['ECG_T_Offsets'] # This is default output of the ecg_process.
    q_indices = q_points[q_points == 1].index.to_list()
    t_indices = t_points[t_points == 1].index.to_list()

    time = np.arange(filtered_signal.size) / fs

    # YOU NEED TO UNCOMMENT THE PLOTTING COMMANDS FOR ANALYSIS TASK 2.  NOTE, if you plot all the
    # figures at the same time, you get memory load warning. You can modify this to plot
    # only few.

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, filtered_signal, label='Filtered ECG Lead II')

    # Calculate QT intervals and plot them as red lines
    qt_intervals = []
    for q, t in zip(q_indices, t_indices):
        if t > q:  # Ensure T point is after Q point for a valid QT interval
            qt_interval = (t - q) / fs  # The indexes are in samples, thus we need to convert them to seconds.
            qt_intervals.append(qt_interval)

            # Plot the QT interval as a red line segment. YOU NEED TO RUN ALSO THESE FOR TASK 2.
            if plot:
                plt.plot([q / fs, t / fs], [filtered_signal[q], filtered_signal[t]], color='red', lw=2,
                     label='QT Interval' if len(qt_intervals) == 1 else "")  # Label only the first for legend clarity

    # YOU NEED TO RUN ALSO THESE FOR TASK 2.
    if plot:
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'{key} - Filtered ECG with QT Intervals')
        plt.legend()
        plt.grid(True)
        #plt.ion()
        plt.show()

    return qt_intervals


# Function to calculate and store average QT interval
def calculateAverageQt(ecg_dict, fs_dict):
    # The average QT intervals for all patients will be stored in average_qt_dict
    average_qt_dict = {}
    i = 0
    for key, ecg_signal in ecg_dict.items():
        i += 1
        if i % 50 == 0 or i == 68 or i == 75 or i == 107:
            plot = True
        else:
            plot = False
        fs = fs_dict[key] # corresponding sampling freq. for each signal
        filtered_signal = filterEcg(ecg_signal, fs) # This calls the filtering function
        qt_intervals = calculateQtIntervals(key, filtered_signal, fs, False) # Calculates the intervals based on the filtered data
        average_qt_interval = np.mean(qt_intervals) if qt_intervals else None # Calculates the average QT for each patient.
        average_qt_dict[key] = average_qt_interval
    return average_qt_dict


# TASK 3: Function to load genotype data and reshape (reshaping to keep the structure for 7 rows). New genotype starts
# every 7th row (if you look at the result txt file, you can see this).

def loadAndReshapeGenotype(filepath):
    results = pd.read_csv(filepath, delimiter="\t", header=None)
    selected = results.iloc[:, 1::7]  # Select every 7th column starting from index 1 --> each new genotype AA, AB or BB.
    s_array = selected.values
    reshaped = s_array.reshape(7, -1)  # Automatically calculate columns based on data size, when we want to have
                                        # the original 7 rows.
    return reshaped


# TASK 3: Function to extract QT intervals based on genotype AA, AB or BB. Thus, this function goes through the reshaped
# data, consisting of 7 rows and 129 columns. 129 is the number of different genotypes i.e. different patients.
# One group to study = same genotype from one row. E.g. all BB genotypes from row 1. YOU NEED THIS INFORMATION ABOUT
# ALL 7 ROWS / GROUPS.

def QtByGenotype(reshaped, average_qt_dict):
    AA_qts_dict = {}
    AB_qts_dict = {}
    BB_qts_dict = {}

    for var_idx, variant in enumerate(reshaped):
        AA_qts_dict[f'Var{var_idx + 1}'] = []
        AB_qts_dict[f'Var{var_idx + 1}'] = []
        BB_qts_dict[f'Var{var_idx + 1}'] = []

        for i in range(len(variant)):
            patient = f'Patient{i + 1}'

            if variant[i] == "AA" and patient in average_qt_dict:
                AA_qts_dict[f'Var{var_idx + 1}'].append(average_qt_dict[patient])
            elif variant[i] == "AB" and patient in average_qt_dict:
                AB_qts_dict[f'Var{var_idx + 1}'].append(average_qt_dict[patient])
            elif variant[i] == "BB" and patient in average_qt_dict:
                BB_qts_dict[f'Var{var_idx + 1}'].append(average_qt_dict[patient])

    return AA_qts_dict, AB_qts_dict, BB_qts_dict


# TASK 3: Function to perform ANOVA and print results. THIS FUNCTION IS COMMENTED OUT, SO YOU CAN FIRST DOWNLOAD YOU DATA.


# def testsForNormality(AAs, ABs, BBs):
#    normality_dict = {
#        "AA" : {},
#        "AB" : {},
#        "BB" : {}
#    }

    # Loop through each variant (AAs)
#    for var_idx, avg_qts in enumerate(AAs):
#        if not avg_qts:
#            normality_dict["AA"][f"var{var_idx + 1}"] = "NA"
#            if len(avg_qts) >= 3:  # Check if there are enough values to perform Shapiro-Wilk test
#                p_value = stat.shapiro(avg_qts).pvalue
#                if p_value > 0.05:
#                    normality_dict["AA"][f"var{var_idx + 1}"] = "Normal"
#                else:
#                    normality_dict["AA"][f"var{var_idx + 1}"] = "Not-Normal"
#            else:
#                normality_dict["AA"][f"var{var_idx + 1}"] = "Not-Normal"

#    return normality_dict


def clean_data(data_list):
    """Convert all values to float and filter out None or non-numeric values."""
    cleaned_data = [float(x) for x in data_list if isinstance(x, (float, int, np.float64))]
    return cleaned_data

def checkGroupsAndTest(AAs, ABs, BBs):
    # Dictionary to store results
    test_results = {}

    # Clean the data for each variant
    for key in AAs.keys():
        AAs[key] = clean_data(AAs[key])
    for key in ABs.keys():
        ABs[key] = clean_data(ABs[key])
    for key in BBs.keys():
        BBs[key] = clean_data(BBs[key])

    # Iterate over each variant (by index)
    for var_idx in range(len(AAs)):
        AA_data = AAs[f"Var{1+var_idx}"]
        AB_data = ABs[f"Var{1+var_idx}"]
        BB_data = BBs[f"Var{1+var_idx}"]

        test_results[f'Var{var_idx + 1}'] = []

        # Create a list to store non-empty groups
        valid_groups = []
        group_names = []

        # Collect non-empty data groups
        if len(AA_data) > 0:
            valid_groups.append(AA_data)
            group_names.append("AA")
        if len(AB_data) > 0:
            valid_groups.append(AB_data)
            group_names.append("AB")
        if len(BB_data) > 0:
            valid_groups.append(BB_data)
            group_names.append("BB")

        # Perform tests based on the number of valid groups
        if len(valid_groups) == 3:
            # Calculate the mean QT intervals for each group
            mean_AA = np.mean(AA_data) if len(AA_data) > 0 else "NA"
            mean_AB = np.mean(AB_data) if len(AB_data) > 0 else "NA"
            mean_BB = np.mean(BB_data) if len(BB_data) > 0 else "NA"

            # Perform Kruskal-Wallis test when all three groups have data
            _, p_value = stats.kruskal(AA_data, AB_data, BB_data, nan_policy="omit")
            test_results[f'Var{var_idx + 1}'].append({
                "test": "Kruskal-Wallis",
                "mean_AA": mean_AA,
                "mean_AB": mean_AB,
                "mean_BB": mean_BB,
                "p_value": p_value
            })

            # Mann-Whitney U tests between AA-BB, AA-AB, AB-BB
            _, p_value = stats.mannwhitneyu(AA_data, BB_data)
            test_results[f'Var{var_idx + 1}'].append({
                "test": "Mann-Whitney U",
                "groups": "AA vs BB",
                "mean_AA": mean_AA,
                "mean_BB": mean_BB,
                "p_value": p_value
            })

            _, p_value = stats.mannwhitneyu(AA_data, AB_data)
            test_results[f'Var{var_idx + 1}'].append({
                "test": "Mann-Whitney U",
                "groups": "AA vs AB",
                "mean_AA": mean_AA,
                "mean_AB": mean_AB,
                "p_value": p_value
            })

            _, p_value = stats.mannwhitneyu(AB_data, BB_data)
            test_results[f'Var{var_idx + 1}'].append({
                "test": "Mann-Whitney U",
                "groups": "AB vs BB",
                "mean_AB": mean_AB,
                "mean_BB": mean_BB,
                "p_value": p_value
            })

        elif len(valid_groups) == 2:
            # Perform Mann-Whitney U test when two groups have data
            group1, group2 = valid_groups
            group1_name, group2_name = group_names

            mean_group1 = np.mean(group1) if len(group1) > 0 else "NA"
            mean_group2 = np.mean(group2) if len(group2) > 0 else "NA"

            _, p_value = stats.mannwhitneyu(group1, group2)
            test_results[f'Var{var_idx + 1}'].append({
                "test": "Mann-Whitney U",
                "groups": f"{group1_name} vs {group2_name}",
                f"mean_{group1_name}": mean_group1,
                f"mean_{group2_name}": mean_group2,
                "p_value": p_value
            })
        else:
            # If there is only one or no valid groups, skip or mark as "NA"
            test_results[f'Var{var_idx + 1}'] = "Not enough data"

    return test_results


# Main processing function
def main():
    id_dict = extract_ids(path_old)
    ecg_dict, field_dict, fs_dict = loadEcg(path_new, id_dict)
    average_qt_dict = calculateAverageQt(ecg_dict, fs_dict)

    # LOAD HERE YOUR GENOTYPE DATA FROM THE RESULTS YOU GOT IN DATA ANALYSIS TASK 1. Use the
    # loadAndReshapeGenotype function, and assign the result to a certain variable.
    genome = loadAndReshapeGenotype("C:/Users/Tomi Viitanen/Desktop/Maisteri opinnot/Periodi I/BINF_1/Project/variant_matches.txt")
    AA_dict, AB_dict, BB_dict = QtByGenotype(genome, average_qt_dict)


    # CALCULATE THE DIFFERENT GROUPS HERE, E.G. FOR GROUP 3 (ROW 3) QT_AB3, QT_BB3, QT_AA3. USE THE
    # QtByGenotype FUNCTION!
    #
    test_results = checkGroupsAndTest(AA_dict, AB_dict, BB_dict)

    for var, tests in test_results.items():
        print(f"Results for {var}:")
        if isinstance(tests, list):
            for test in tests:
                if isinstance(test, dict):
                    print(f"  Test: {test['test']}")
                    if "groups" in test:
                        print(f"    Groups: {test['groups']}")
                    if "mean_AA" in test:
                        print(f"    Mean AA: {test['mean_AA']:.5f}")
                    if "mean_AB" in test:
                        print(f"    Mean AB: {test['mean_AB']:.5f}")
                    if "mean_BB" in test:
                        print(f"    Mean BB: {test['mean_BB']:.5f}")
                    print(f"    p-value: {test['p_value']:.5f}")
                else:
                    print(f"  {test}")
        else:
            print(f"  {tests}")
        print()


    val_list = []
    for pat, val in average_qt_dict.items():
        print(pat, val.item())
        val_list.append(val)

    print("Variance: ", np.std(val_list))
    print("Mean: ", np.mean(val_list))
    print("min: ", np.min(val_list))
    print("max: ", np.max(val_list))


# Run the main function
if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import h5py
import matplotlib
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow.python.keras.models import model_from_json
from ImSetNorm import Normalize
import warnings

warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")


class RbRatioAnalyses:
    """
        A class for analyzing radial bias ratios in response to sinusoidal grating stimuli
        of various spatial frequencies and orientations using different neural models.

        Attributes:
            root (str): Root directory for the project.
            cnn_model_root (str): Directory for CNN model.
            rsl_model_root (str): Directory for RSL model.
            stim_root (str): Root directory for stimuli.
            cnn_stim (str): Subdirectory for undistorted stimuli for CNN.
            rsl_stim (str): Subdirectory for distorted stimuli for RSL.
            cnn_model: Loaded CNN model.
            rsl_model: Loaded RSL model.
            fmap_s (int): Feature map size.
            FOV (int): Field of view.
            eccentricity (int): Eccentricity of the stimuli.
            angles: Array of polar angles.
            rad: Array of polar radii.
            population_maps_cnn: Population maps for CNN model.
            orientation_maps_cnn: Orientation maps for CNN model.
            population_maps_rsl: Population maps for RSL model.
            orientation_maps_rsl: Orientation maps for RSL model.
            bin_activities_cnn: Binned activities for CNN model.
            bin_activities_rsl: Binned activities for RSL model.
            ratios_cnn: Radial bias ratios for CNN model.
            ratios_rsl: Radial bias ratios for RSL model.
            sfs (list): List of spatial frequencies.
            cpi: Cycles per degree for each spatial frequency.
            ori_folders (list): List of orientation folders.
            bin_orientations (list): List of bin orientations.
            bins: Bins for radial analysis.
            bin_areas: Areas for radial analysis.
            grating_ang (list): List of grating orientations.
            grating_ang_corr (list): Corrected list of grating orientations.

        Methods:
            __init__(self, fov=20, fmap_s=128): Constructor for the class.
            load_model(self, model): Load a model from JSON and weights files.
            load_stim(self, model, sf, ori, splitter='_'): Load sinusoidal grating stimuli images.
            reshape_map(matrix): Reshape a feature map matrix.
            create_polar_coordinates(self): Create polar coordinates for stimuli analysis.
            save_model_results(self, model_name=None, populationmaps=None, orientationmaps=None): Save model results.
            save_bin_activities(self, model_name=None, bin_activities=None): Save binned activities.
            load_bin_activities(self, model_name=None): Load binned activities from file.
            load_model_results(self, model_name=None, orientation=1, population=0): Load model results.
            load_rb_ratios(self, model_name=None): Load radial bias ratios from file.
            model_responses(self, model_name): Compute model responses to stimuli.
            create_radial_bins(self): Create bins for radial analysis.
            get_bin_activities(self, model_name): Calculate binned activities.
            rb_ratio(self, model_name): Compute radial bias ratios.
            get_ratios(self, c_deg, ori, model_name): Get radial bias ratios for specific conditions.
            rb_ratio_graph(self, model_name): Create and save a graph of radial bias ratios.
    """
    def __init__(self, fov=20, fmap_s=128):
        """
            Constructor for the RbRatioAnalyses class.

            Args:
                fov (int): Field of view for stimuli analysis.
                fmap_s (int): Size of feature maps.
        """
        self.root = "H:/RSLprojectLukas/RadialBias/"
        self.cnn_model_root = "CNN_model"
        self.rsl_model_root = "RSL_model"

        self.stim_root = "stimuli/RBRatio/"
        self.cnn_stim = "0_undistorted/"
        self.rsl_stim = "1_distorted/"

        self.cnn_model = self.load_model("cnn")
        self.rsl_model = self.load_model("rsl")
        self.fmap_s = fmap_s
        self.FOV = fov
        self.eccentricity = int(self.FOV / 2)
        self.angles = None
        self.rad = None
        self.create_polar_coordinates()

        self.population_maps_cnn = None
        self.orientation_maps_cnn = None
        self.population_maps_rsl = None
        self.orientation_maps_rsl = None
        self.bin_activities_cnn = None
        self.bin_activities_rsl = None
        self.ratios_cnn = None
        self.ratios_rsl = None

        self.sfs = [2, 5, 10, 20, 40, 60, 80, 100, 120]
        self.cpi = np.array(self.sfs) / self.FOV
        self.ori_folders = ["1_-67.5/", "2_-45.0/", "3_-22.5/", "4_0.0/", "5_22.5/", "6_45.0/", "7_67.5/", "8_90.0/"]
        self.bin_orientations = [-180.0, -157.5, -135.0, -112.5, -90.0, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90.0,
                                 112.5, 135.0, 157.5, -180]
        self.frequencies = {"2": 0, "5": 1, "10": 2, "20": 3, "40": 4, "60": 5, "80": 6, "100": 7, "120":8}
        self.bins = None
        self.bin_areas = None

        self.grating_ang = [-67.5, -45.0, -22.5, 0.0, 22.5, 45.0, 67.5, 90.0]
        self.grating_ang_corr = [22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, -180.0]

    def load_model(self, model):
        """
            Load a neural model from JSON and weights files.

            Args:
                model (str): Model identifier ("cnn" for CNN, "rsl" for RSL).

            Returns:
                keras.Model: Loaded neural model.
        """
        if model == "cnn":
            folder = self.cnn_model_root
            print("Loading standard CNN Model")
        elif model == "rsl":
            folder = self.rsl_model_root
            print("Loading RSL Model")
        else:
            print("Invalid model name")
            exit()
        json_file = open(folder + '/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(folder + "/model.h5")
        print("Loaded model from drive")
        loaded_model = keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[0].output)
        return loaded_model

    def load_stim(self, model, sf, ori, splitter='_'):
        """
            Load sinusoidal grating stimuli images for a specific model, spatial frequency, and orientation.

            Args:
                model (str): Model identifier ("cnn" for CNN, "rsl" for RSL).
                sf (int): Spatial frequency in cycles per image.
                ori (str): Orientation folder name.
                splitter (str, optional): Splitter used for file names. Default is '_'.

            Returns:
                numpy.ndarray: Loaded stimuli images.
        """
        try:
            if model == "cnn":
                folder = f"{self.stim_root}{self.cnn_stim}{sf}/{ori}"
                print(f"Loading {sf} cpi, {sf/20} c/deg stimuli, with orientation {ori} for CNN Model")
            elif model == "rsl":
                folder = f"{self.stim_root}{self.rsl_stim}{sf}/{ori}"
                print(f"Loading {sf} cpi, {sf/20} c/deg, with orientation {ori} stimuli for RSL Model")
        except:
            print("Invalid model name")
            exit()
        files = os.listdir(folder)
        files = [f.split(splitter) for f in files]
        for i, e in enumerate(files):
            files[i][0] = int(files[i][0])
        files.sort()
        for i, e in enumerate(files):
            files[i][0] = str(files[i][0])
        files = [splitter.join(f) for f in files]
        images = np.asarray([cv2.imread(folder + file, 1) for file in files])
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 3)
        # images = images.reshape(images.shape[0], images.shape[1] * images.shape[2], 3)
        images = images.astype('float32')
        return images

    @staticmethod
    def reshape_map(matrix):
        """
            Reshape a feature map matrix.

            Args:
                matrix (numpy.ndarray): Input feature map matrix.

            Returns:
                numpy.ndarray: Reshaped feature map.
        """
        output = np.asarray([matrix[0, :, :, i] for i in range(matrix.shape[3])])
        return output

    def create_polar_coordinates(self):
        """
            Create polar coordinates for stimuli analysis.
        """
        t = np.linspace(-self.eccentricity, self.eccentricity, 128)
        x, y = np.meshgrid(t, t)
        y = -y
        self.angles = np.rad2deg(np.angle(x + y * 1j))
        self.rad = np.abs(x + y * 1j)

    def save_model_results(self, model_name=None, populationmaps=None, orientationmaps=None):
        """
            Save model results such as population and orientation maps.

            Args:
                model_name (str, optional): Model name ("cnn" for CNN, "rsl" for RSL).
                populationmaps (numpy.ndarray, optional): Population maps data.
                orientationmaps (numpy.ndarray, optional): Orientation maps data.
        """
        if not populationmaps and not orientationmaps:
            if model_name:
                if model_name == "cnn":
                    if type(self.population_maps_cnn) is np.ndarray:
                        populationmaps = self.population_maps_cnn
                        orientationmaps = self.orientation_maps_cnn
                    else:
                        print("No featuremaps for cnn model in memory")
                        exit()
                elif model_name == "rsl":
                    if type(self.population_maps_rsl) is np.ndarray:
                        populationmaps = self.population_maps_rsl
                        orientationmaps = self.orientation_maps_rsl
                    else:
                        print("No featuremaps for rsl model in memory")
                        exit()
                else:
                    print("Invalid model name")
                    exit()
            else:
                print("No modelname was found")
                exit()
        if model_name:
            try:
                print(f"Saving population maps for {model_name} model")
                with h5py.File("H:/RSLprojectLukas/rb_results/population_maps_{}".format(model_name), 'w') as f:
                    f.create_dataset("population_maps", data=populationmaps)
                    f.close()
                print(f"Saving orientation maps for {model_name} model")
                with h5py.File("H:/RSLprojectLukas/rb_results/orientation_maps_{}".format(model_name), 'w') as f:
                    f.create_dataset("orientation_maps", data=orientationmaps)
                    f.close()
            except:
                print("Error in save data")
        else:
            print("No modelname was found")

    def save_bin_activities(self, model_name=None, bin_activities=None):
        """
            Save binned activities for a specific model.

            Args:
                model_name (str, optional): Model name ("cnn" for CNN, "rsl" for RSL).
                bin_activities (numpy.ndarray, optional): Binned activities data.
        """
        if model_name:
            if not bin_activities:
                if model_name == "cnn":
                    if self.bin_activities_cnn is not None:
                        bin_activities = self.bin_activities_cnn
                    else:
                        print("No bin activities for cnn model in memory")
                        exit()
                elif model_name == "rsl":
                    if self.bin_activities_rsl is not None:
                        bin_activities = self.bin_activities_rsl
                    else:
                        print("No featuremaps for rsl model in memory")
                        exit()
                else:
                    print("Invalid model name")
                    exit()
                try:
                    print(f"Saving bin activity for {model_name} model")
                    with h5py.File("H:/RSLprojectLukas/rb_results/bin_activities_{}".format(model_name), 'w') as f:
                        f.create_dataset("bin_activity", data=bin_activities)
                        f.close()
                except:
                    print("Error in save data")
        else:
            print("No modelname was found")
            exit()

    def load_bin_activities(self, model_name=None):
        """
            Load binned activities data for a specific model.

            Args:
                model_name (str, optional): Model name ("cnn" for CNN, "rsl" for RSL).

            Returns:
                numpy.ndarray: Loaded binned activities data.
        """
        bin_activities = None
        try:
            print(f"Loading population maps for {model_name} model")
            with h5py.File("H:/RSLprojectLukas/rb_results/bin_activities_{}".format(model_name), 'r') as f:
                if model_name == "cnn":
                    self.bin_activities_cnn = f.get("bin_activity")[()]
                    f.close()
                elif model_name == "rsl":
                    self.bin_activities_rsl = f.get("bin_activity")[()]

                else:
                    bin_activities = f.get("bin_activity")[()]
                f.close()
            print(f"Bin activities for {model_name} model loaded")
        except:
            print(f"No bin activity data found for {model_name} model\n")
        if bin_activities:
            return bin_activities

    def load_model_results(self, model_name=None, orientation=1, population=0):
        """
            Load model results such as population and orientation maps.

            Args:
                model_name (str, optional): Model name ("cnn" for CNN, "rsl" for RSL).
                orientation (int, optional): Flag for loading orientation maps. Default is 1.
                population (int, optional): Flag for loading population maps. Default is 0.
        """
        if model_name == "cnn" or model_name == "rsl":
            if population:
                try:
                    print(f"Loading population maps for {model_name} model")
                    with h5py.File("H:/RSLprojectLukas/rb_results/population_maps_{}".format(model_name), 'r') as f:
                        if model_name == "cnn":
                            self.population_maps_cnn = f.get("population_maps")[()]
                        elif model_name == "rsl":
                            self.population_maps_rsl = f.get("population_maps")[()]
                        f.close()
                    print(f"Population maps for {model_name} model loaded")
                except:
                    print(f"No population map data found for {model_name}model\n")
            if orientation:
                try:
                    print(f"Loading orientation maps for {model_name} model")
                    with h5py.File("H:/RSLprojectLukas/rb_results/orientation_maps_{}".format(model_name), 'r') as f:
                        if model_name == "cnn":
                            self.orientation_maps_cnn = f.get("orientation_maps")[()]
                        elif model_name == "rsl":
                            self.orientation_maps_rsl = f.get("orientation_maps")[()]
                        f.close()
                    print(f"Orientation maps for {model_name} model loaded\n")
                except:
                    print(f"No orientation map data found for {model_name}model")
        else:
            print("Invalid model name")
            exit()

    def load_rb_ratios(self, model_name=None):
        """
            Load radial bias ratios data for a specific model.

            Args:
                model_name (str, optional): Model name ("cnn" for CNN, "rsl" for RSL).
        """
        if model_name == "cnn" or model_name == "rsl":
            try:
                print(f"Loading radial bias ratios for {model_name} model")
                with h5py.File("H:/RSLprojectLukas/rb_results/rb_ratios_{}".format(model_name), 'r') as f:
                    if model_name == "cnn":
                        self.ratios_cnn = f.get("rb_ratios")[()]
                    elif model_name == "rsl":
                        self.ratios_rsl = f.get("rb_ratios")[()]
                    f.close()
                print(f"Radial bias ratios for {model_name} model loaded")
            except:
                print(f"No population map data found for {model_name}model\n")
        else:
            print("Invalid model name")
            exit()

    def model_responses(self, model_name):
        """
            Compute model responses to stimuli for a specific model.

            Args:
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
        """
        if model_name == "cnn":
            model = self.cnn_model
            print("Getting model responses for the CNN model")
        elif model_name == "rsl":
            model = self.rsl_model
            print("Getting model responses for the RSL model")
        else:
            print("Invalid model name")
            exit()
        population_maps_all = []
        orientation_maps_all = []
        for i, sf in enumerate(self.sfs):
            population_maps_sf = []
            orientation_maps_sf = []
            for j, ori in enumerate(self.ori_folders):
                population_map_ori = []
                stim = self.load_stim(model_name, sf, ori)
                stim = np.asarray(Normalize(stim, 1))
                for k, s in enumerate(stim):
                    img = np.expand_dims(s[:, :, :], axis=0)
                    feature_maps = model.predict(img)
                    feature_map = (self.reshape_map(feature_maps))
                    population_map_ori.append(np.array(np.mean(feature_map, axis=0)))
                population_maps_sf.append(np.array(population_map_ori))
                orientation_maps_sf.append(np.array(np.mean(population_map_ori, axis=0)))
            population_maps_all.append(np.array(population_maps_sf))
            orientation_maps_all.append(np.array(orientation_maps_sf))
        if model_name == "cnn":
            self.population_maps_cnn = np.array(population_maps_all)
            self.orientation_maps_cnn = np.array(orientation_maps_all)
        elif model_name == "rsl":
            self.population_maps_rsl = np.array(population_maps_all)
            self.orientation_maps_rsl = np.array(orientation_maps_all)

    def create_radial_bins(self):
        """
            Create bins for radial analysis.
        """
        sum, area = [], []
        for o in self.bin_orientations:
            if o == -180:
                lower = o + 22.5
                upper = -o - 22.5
                msk1 = True * ((self.angles < lower) + (self.angles > upper))
                msk2 = True * (self.rad <= 9)
            else:
                lower = o - 22.5
                upper = o + 22.5
                msk1 = True * ((self.angles > lower) & (self.angles < upper))
                msk2 = True * (self.rad <= 9)
            msk = msk1 * msk2
            sum.append(np.sum(msk))
            area.append(msk)
        self.bins = area
        self.bin_areas = sum

    def get_bin_activities(self, model_name):
        """
            Calculate binned activities for a specific model.

            Args:
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
        """
        if model_name == "cnn":
            orientationmaps = self.orientation_maps_cnn
            print("Loading orientation maps for the CNN model")

        elif model_name == "rsl":
            orientationmaps = self.orientation_maps_rsl
            print("Loading orientation maps for the RSL model")
        else:
            print("Invalid model name")
            exit()
        activations = []
        for i, sf in enumerate(self.sfs):
            sf_activity = []
            for j, gating_ori in enumerate(self.grating_ang):
                ang_activity = []
                for k, o in enumerate(self.bin_orientations):
                    msk = self.bins[k]
                    t = orientationmaps[i][j] * msk
                    ang_activity.append(np.divide(np.sum(t), self.bin_areas[k]))
                sf_activity.append(np.array(ang_activity))
            activations.append(np.array(sf_activity))
        bin_activities = np.array(activations)
        if model_name == "cnn":
            self.bin_activities_cnn = bin_activities
        elif model_name == "rsl":
            self.bin_activities_rsl = bin_activities
        print(f"Bin activities for {model_name} model stored\n")

    def rb_ratio(self, model_name):
        """
            Compute radial bias ratios for a specific model.

            Args:
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
        """
        if model_name == "cnn":
            data = self.bin_activities_cnn
            print("\nLoading bin activities for the CNN model")

        elif model_name == "rsl":
            data = self.bin_activities_rsl
            print("\nLoading bin activities for the RSL model")
        else:
            print("Invalid model name")
            exit()
        n_bin = np.shape(self.bin_orientations)[0]
        cutoff = n_bin / 2
        rb_ratios = []
        for i, sf in enumerate(self.sfs):
            rb_per_ori = []
            for j, grat_ori in enumerate(self.grating_ang_corr):
                theta_index = np.where(np.array(self.bin_orientations) == grat_ori)[0][0]
                if theta_index > cutoff:
                    theta_opp = int(theta_index - n_bin / 2)
                    theta_orth = int(theta_index - n_bin / 4)
                else:
                    theta_opp = int(theta_index + n_bin / 2)
                    theta_orth = int(theta_index + n_bin / 4)
                if theta_orth < cutoff:
                    theta_orth_opp = int(theta_orth + n_bin / 2)
                else:
                    theta_orth_opp = int(theta_orth - n_bin / 2)
                #print(theta_index, theta_opp, theta_orth, theta_orth_opp)
                rb_ratio = (data[i, j, theta_index] + data[i,j, theta_opp]) / (data[i, j, theta_orth] + data[i, j, theta_orth_opp])
                rb_per_ori.append(rb_ratio)
            rb_ratios.append(np.array(rb_per_ori))
        if model_name == "cnn":
            print("Storing radial bias ratios for the CNN model\n")
            self.ratios_cnn = np.array(rb_ratios)
        elif model_name == "rsl":
            print("Storing radial bias ratios for the RSL model\n")
            self.ratios_rsl = np.array(rb_ratios)
        print(f"Saving bin activity for {model_name} model")
        with h5py.File("H:/RSLprojectLukas/rb_results/rb_ratios_{}".format(model_name), 'w') as f:
            f.create_dataset("rb_ratios", data=rb_ratios)
            f.close()

    def get_ratios(self, c_deg, ori, model_name):
        """
            Get radial bias ratios for specific conditions.

            Args:
                c_deg (float or list of float): Spatial frequency in cycles per degree.
                ori (float or list of float): Grating orientation in degrees.
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
        """
        if model_name == "cnn":
            ratio_data = self.ratios_cnn
            full_name = "CORnet-Z"
            print("\nLoading bin activities for the CNN model")
        elif model_name == "rsl":
            ratio_data = self.ratios_rsl
            full_name = "RSL-CORnet-Z"
            print("\nLoading bin activities for the RSL model")
        else:
            print("Invalid model name")
            exit()

        if isinstance(c_deg, list):
            pass
        elif isinstance(c_deg, int) or isinstance(c_deg, float):
            c_deg = [c_deg]
        else:
            print("Incorrect spatial frequencies entered. Please enter a list of spatial frequencies in cycles/degree")
            exit()

        if isinstance(ori, list):
            pass
        elif isinstance(ori, int) or isinstance(ori, float):
            ori = [ori]
        else:
            print("Incorrect stimulus orientations entered. Please enter a list of grating orientations in degrees")
            exit()

        for sf in c_deg:
            try:
                sf_ind = np.where(self.cpi == sf)[0][0]
            except:
                print("Incorrect spatial frequencies entered. Please enter a list of spatial frequencies in cycles/degree")
                exit()
            for o in ori:
                flag = 0
                try:
                    or_ind = np.where(np.array(self.grating_ang_corr) == o)[0][0]
                    flag = 1
                except:
                    pass
                if flag == 0:
                    try:
                        or_ind = np.where(np.array(self.grating_ang_corr) == o-180)[0][0]
                    except:
                        print("Incorrect stimulus orientations entered. Please enter a list of grating orientations in degrees")
                        exit()
                ratio = ratio_data[sf_ind, or_ind]
                print(f"Radial Bias ratio for stimuli of {sf} c/deg, and an orientation of {o} is: {ratio}")

    @staticmethod
    def axes_ratio(data, stim_o, bin_o):
        """
            Calculate the radial bias ratio for a given stimulus orientation.

            This method calculates the radial bias ratio based on the provided data,
            stimulus orientation, and bin orientations.

            Args:
                data (numpy.ndarray): Array of data containing radial bias values.
                stim_o (float): Stimulus orientation in degrees.
                bin_o (numpy.ndarray): Array of bin orientations in degrees.

            Returns:
                float: Radial bias ratio for the specified stimulus orientation.
        """
        n_bin = np.shape(bin_o)[0]
        cutoff = n_bin / 2
        theta_stim = stim_o
        theta_index = np.where(np.array(bin_o) == theta_stim)[0][0]
        if theta_index > cutoff:
            theta_opp = int(theta_index - n_bin / 2)
            theta_orth = int(theta_index - n_bin / 4)
        else:
            theta_opp = int(theta_index + n_bin / 2)
            theta_orth = int(theta_index + n_bin / 4)
        if theta_orth < cutoff:
            theta_orth_opp = int(theta_orth + n_bin / 2)
        else:
            theta_orth_opp = int(theta_orth - n_bin / 2)
        print(theta_index, theta_opp, theta_orth, theta_orth_opp)
        rb_ratio = (data[theta_index] + data[theta_opp]) / (data[theta_orth] + data[theta_orth_opp])
        return rb_ratio

    def create_polar_plots(self, model_name, frequency_list):
        """
            Create polar plots for visualizing radial bias across orientations and spatial frequencies.

            This method prepares the data needed for generating polar plots to visualize the radial bias for different orientations
            and spatial frequencies of a given model.

            Args:
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
                frequency_list (list): List of spatial frequencies to create polar plots for.
        """
        if model_name == "cnn":
            bin_activities = list(self.bin_activities_cnn)
        elif model_name == "rsl":
            bin_activities = list(self.bin_activities_rsl)

        self.bin_orientations.append(self.bin_orientations[0])
        self.bin_orientations_rad = np.deg2rad(self.bin_orientations)

        for sf in frequency_list:
            key = self.frequencies[str(sf)]
            sf_bin_data = list(bin_activities[key])

            print("What you need to know: ", np.shape(sf_bin_data))

            act_min = np.min([np.min(a) for a in sf_bin_data])
            act_max = np.max([np.max(a) for a in sf_bin_data])
            for ind, a in enumerate(sf_bin_data):
                sf_bin_data[ind] = np.append(sf_bin_data[ind], a[0])

            act_set1 = [sf_bin_data[1], sf_bin_data[5]]
            act_set2 = [sf_bin_data[7], sf_bin_data[3]]
            act_set3 = [sf_bin_data[0], sf_bin_data[4]]
            act_set4 = [sf_bin_data[2], sf_bin_data[6]]

            act_set1[0] = act_set1[0] / np.max(act_set1[0])
            act_set1[1] = act_set1[1] / np.max(act_set1[1])
            act_set2[0] = act_set2[0] / np.max(act_set2[0])
            act_set2[1] = act_set2[1] / np.max(act_set2[1])

            act_set3[0] = act_set3[0] / np.max(act_set3[0])
            act_set3[1] = act_set3[1] / np.max(act_set3[1])
            act_set4[0] = act_set4[0] / np.max(act_set4[0])
            act_set4[1] = act_set4[1] / np.max(act_set4[1])

            self.create_polar_graph(act_set1, model_name, ['45_degrees', '135_degrees'], sf)
            self.create_polar_graph(act_set2, model_name, ['0_degrees', '90_degrees'], sf)

    def create_polar_graph(self, data, model_name, title, sf):
        """
            Plots the polar graph to visualize radial bias data for a specific condition.

            This method generates a polar graph to visualize the radial bias data for a specific
            condition, including orientation and spatial frequency.

            Args:
                data (list of numpy.ndarray): List of radial bias data for different orientations.
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
                title (list of str): Titles for different orientations.
                sf (float): Spatial frequency in cycles per degree.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 2))  # For RSL
        matplotlib.rcParams['font.family'] = 'sans-serif'
        ax1 = fig.add_axes([0.15, 0.5, 0, 0.4], polar=False)  # 0.125, 0.5, 0, 0.4 #0.10, 0.5, 0, 0.4
        ax1.set_xticks([])
        ax1.set_yticks([0.0, 1.0])
        foo = ['0', '1']
        ax1.set_yticklabels(foo)
        ax1.tick_params('both', length=00, width=0, which='major', labelsize=5)
        ax1.tick_params('both', length=00, width=0, which='minor')
        plt.ylim(0, 1)
        d1 = ax.plot(self.bin_orientations_rad, data[0], marker='.', markersize=1, ls='-', linewidth=1.5)
        d2 = ax.plot(self.bin_orientations_rad, data[1], marker='.', markersize=1, ls='-', linewidth=1.5)
        ax.legend([d1[0], d2[0]], title, framealpha=1, markerscale=1, loc='lower right', bbox_to_anchor=(1.5, -0.18),
                  frameon=False, fontsize=7)  # for main figure: bbox_to_anchor = (1.38, -0.15) #fontsize for RSL = 5
        ax.set_rmin(0)
        ax.set_rmax(1)
        ax.set_yticklabels([])
        fig.subplots_adjust(wspace=0.5)
        ax.tick_params(axis="both", labelsize=5, width=1.5)
        ax1.tick_params(axis="both", labelsize=5, width=1.5)
        plt.ylabel("Averaged Cell Activity", fontsize=7, labelpad=0)
        ax.tick_params(axis="x", pad=-4)  # RSL pad = -4
        plt.savefig(f"results/radialBias/{model_name}/RB_polar_{sf}_{title[0]}_{title[1]}.svg", dpi=600,
                    format='svg')

    def rb_ratio_graph(self, model_name):
        """
            Create and save a graph of radial bias ratios for a specific model.

            Args:
                model_name (str): Model name ("cnn" for CNN, "rsl" for RSL).
        """
        if model_name == "cnn":
            ratio_data = self.ratios_cnn
            graph_color = '#EAC152'
            full_name = "CORnet-Z"
            print("Loading radial bias ratios for the CNN model")
        elif model_name == "rsl":
            ratio_data = self.ratios_rsl
            graph_color = '#91C4FF'
            full_name = "RSL-CORnet-Z"
            print("Loading radial bias ratios for the RSL model")
        else:
            print("Invalid model name")
            exit()
        print("Creating graph")
        mean_ratios = np.array([np.mean(ratio_sf) for ratio_sf in ratio_data])
        ratios_err = np.array([stats.sem(ratio_sf) for ratio_sf in ratio_data])

        width = 1.5
        fig, ax = plt.subplots(figsize=(3.3, 2.75))
        matplotlib.rcParams['font.family'] = 'sans-serif'
        ax.plot(self.cpi[:-1], mean_ratios[:-1], marker=None, ls='-', color=graph_color, linewidth=width,
                      label=full_name)
        ax.fill_between(self.cpi[:-1], mean_ratios[:-1] - (ratios_err[:-1] * 1.96), mean_ratios[:-1] +
                        (ratios_err[:-1] * 1.96), alpha=0.4, color=graph_color)
        ax.plot(self.cpi[:-1], [1, 1, 1, 1, 1, 1, 1, 1], marker=None, ls='--',
                       color='gray', linewidth=0.5)
        ax.legend(framealpha=1, markerscale=1, loc='lower right', frameon=False, fontsize=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlim(0, 5)
        plt.ylim(0.5, 1.5)
        ax.minorticks_off()
        ax.tick_params('both', length=0, width=0, which='minor')
        plt.locator_params(axis="x", nbins=5)
        plt.locator_params(axis="y", nbins=2)
        ax.set_xticklabels(('0', '', '', '', '', '5'), fontsize=7)
        ax.set_yticklabels(('0.5', '1', '1.5'), fontsize=7)
        plt.xlabel("Spatial Frequency (c/deg)", family='sans-serif', size=7)
        plt.ylabel("Radial Bias Ratio", family='sans-serif', size=7)
        plt.tight_layout()
        plt.savefig(f"results/radialBias/{model_name}/Radial_Bias_ratios_{model_name}.svg", dpi=600, format='svg')


if __name__ == '__main__':
    experiment = RbRatioAnalyses(fov=20)

    load_responses = 0
    get_bin_activities = 1
    get_rb_ratios = 1
    rb_sf_graph = 1

    if load_responses:
        experiment.load_model_results("cnn")
        experiment.load_model_results("rsl")

    else:
        experiment.model_responses("cnn")
        experiment.model_responses("rsl")
        experiment.save_model_results("cnn")
        experiment.save_model_results("rsl")

    if get_bin_activities:
        experiment.create_radial_bins()
        experiment.get_bin_activities("cnn")
        experiment.get_bin_activities("rsl")
        experiment.save_bin_activities("cnn")
        experiment.save_bin_activities("rsl")

    else:
        experiment.load_bin_activities("cnn")
        experiment.load_bin_activities("rsl")

    if get_rb_ratios:
        experiment.rb_ratio("cnn")
        experiment.rb_ratio("rsl")

    else:
        experiment.load_rb_ratios("cnn")
        experiment.load_rb_ratios("rsl")

    experiment.create_polar_plots("rsl", [2, 5, 10, 20, 40, 60, 80, 100, 120])

    if rb_sf_graph:
        experiment.rb_ratio_graph("cnn")
        experiment.rb_ratio_graph("rsl")

    experiment.get_ratios(4, [45.0, 135.0, 0.0, 90.0], "cnn")
    experiment.get_ratios(4, [45.0, 135.0, 0.0, 90.0], "rsl")

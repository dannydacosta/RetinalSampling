import cv2
import numpy as np
import os
import h5py
import RSL as RSL_File
import matplotlib
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow.python.keras.models import model_from_json
from ImSetNorm import Normalize
import warnings
import copy
import sys

warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")


class PrfMapping:
    """
    A class for performing Population Receptive Field (pRF) mapping analysis on convolutional neural network models.

    This class encapsulates methods for loading pre-trained convolutional neural network models, processing stimulus
    images, calculating response profiles, and generating pRF maps. It facilitates the analysis of pRF properties, such as
    eccentricity, orientation, and size, using the model's responses to stimuli.

    Attributes:
        fov (int): Field of view for the pRF mapping, default is 20.
        root (str): Root directory for data storage and retrieval.
        cnn_model_root (str): Directory containing CNN model files.
        rsl_model_root (str): Directory containing RSL model files.
        rsl_model_cut (tensorflow.keras.Model): Cut RSL model for specific layers.
        cnn_model_cut (tensorflow.keras.Model): Cut CNN model for specific layers.
        stim_root (str): Root directory for stimulus images.
        cnn_stim_folder (str): Subfolder for undistorted stimulus images for CNN.
        rsl_stim_folder (str): Subfolder for distorted stimulus images for RSL.
        sub_folders (list): List of subfolders for different bar orientations.
        layers (list): List of neural network layers for analysis.
        outsize (dict): Dictionary containing output sizes for different layers.
        rsl_result_folder (str): Directory for RSL model result storage.
        cnn_result_folder (str): Directory for CNN model result storage.
        batch: Placeholder for batch of stimulus images.
        steps_line (int): Number of steps for horizontal/vertical lines.
        steps_diagonal (int): Number of steps for diagonal lines.
        total_steps (int): Total number of steps.
        cnn_model (tensorflow.keras.Model): Loaded CNN model.
        rsl_model (tensorflow.keras.Model): Loaded RSL model.
        cnn_response_profile (dict): Dictionary storing response profiles for CNN model.
        rsl_response_profile (dict): Dictionary storing response profiles for RSL model.
        active_response_profile (dict): Dictionary storing active response profiles for the current model.
        cnn_prf_maps (dict): Dictionary storing pRF maps for CNN model.
        rsl_prf_maps (dict): Dictionary storing pRF maps for RSL model.
        gauss_responses: Placeholder for Gaussian response profiles.
        FOV (int): Field of view for the pRF analysis.
        eccentricity (int): Eccentricity for the pRF analysis.
        angles: Placeholder for angles for the pRF analysis.
        rad: Placeholder for radius values for the pRF analysis.
    """
    def __init__(self, fov=20):
        self.root = "H:/RSLprojectLukas/RadialBias/"
        self.cnn_model_root = "CNN_model"
        self.rsl_model_root = "RSL_model"
        self.rsl_model_cut = None
        self.cnn_model_cut = None

        self.stim_root = "stimuli/pRF/"
        self.cnn_stim_folder = "0_undistorted/"
        self.rsl_stim_folder = "1_distorted/RSL/"
        self.sub_folders = ["0_0/", "1_90/", "2_45/", "3_135/"]
        self.layers = ["V1", "V2", "V4"]
        self.outsize = {"V1": 128, "V2": 64, "V4": 32}
        self.rsl_result_folder = "results/prf/rsl/"
        self.cnn_result_folder = "results/prf/cnn/"

        self.batch = None
        self.steps_line = 65  # 65
        self.steps_diagonal = 92
        self.total_steps = 2 * self.steps_line + 2 * self.steps_diagonal

        self.cnn_model = self.load_model("cnn")
        self.rsl_model = self.load_model("rsl")

        self.cnn_response_profile = {"V1": None, "V2": None, "V4": None}
        self.rsl_response_profile = {"V1": None, "V2": None, "V4": None}
        self.active_response_profile = {"V1": None, "V2": None, "V4": None}

        self.cnn_prf_maps = {"V1": None, "V2": None, "V4": None}
        self.rsl_prf_maps = {"V1": None, "V2": None, "V4": None}

        self.gauss_responses = None

        self.FOV = fov
        self.eccentricity = int(self.FOV / 2)
        self.angles = None
        self.rad = None

    def load_model(self, model):
        """
        Loads a Keras model from a JSON file and its corresponding weights.

        Args:
            model (str): Model name ("cnn" or "rsl").

        Returns:
            loaded_model: The loaded Keras model.
        """
        if model == "cnn":
            folder = self.cnn_model_root
            print("Loading standard CNN Model")
            self.current_model_name = "cnn"
        elif model == "rsl":
            folder = self.rsl_model_root
            print("Loading RSL Model")
            self.current_model_name = "rsl"
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
        return loaded_model

    def cut_model(self, layer):
        """
        Creates a new Keras model by cutting a specific layer from the loaded model.

        Args:
            layer (int): Index of the layer to cut.
        """
        if self.current_model_name == "cnn":
            c_model = copy.copy(self.cnn_model)
            print(f"Cutting CNN model at layer {self.layer_name}")
            self.cnn_model_cut = keras.Model(inputs=c_model.input, outputs=c_model.layers[layer].output)
        elif self.current_model_name == "rsl":
            c_model = copy.copy(self.rsl_model)
            print(f"Cutting RSL model at layer {self.layer_name}")
            self.rsl_model_cut = keras.Model(inputs=c_model.input, outputs=c_model.layers[layer].output)

    @staticmethod
    def reshape_matrix(matrix):
        """
        Reshapes a matrix to a specific format.

        Args:
            matrix (numpy.ndarray): Input matrix.

        Returns:
            numpy.ndarray: Reshaped matrix.
        """
        output = np.asarray([matrix[0, :, :, i] for i in range(matrix.shape[3])])
        return output

    @staticmethod
    def running_mean(mo, v, n):
        """
        Calculates a running mean based on previous mean, new value, and count.

        Args:
            mo (numpy.ndarray): Previous mean.
            v (numpy.ndarray): New value.
            n (int): Count.

        Returns:
            numpy.ndarray: Updated mean.
        """
        if n == 1:
            mn = v
        else:
            mn = np.add(mo, (np.subtract(v, mo) / n))
        return mn

    @staticmethod
    def correlation(A, B):
        """
        Computes the correlation between a vector A and each row of matrix B.

        A is a vector of length n and B an m-by-n matrix.
        This function computes the correlation between A and each row of B,
        thus returning m correlations

        Args:
            A (numpy.ndarray): Vector of length n.
            B (numpy.ndarray): m-by-n matrix.

        Returns:
            numpy.ndarray: Array of correlations.
        """
        mag_A = np.sqrt(np.dot(A, A))
        mag_B = np.sqrt(np.sum(B ** 2, axis=1))
        return np.matmul(B, A) / (mag_A * mag_B)

    def white_mask(self, im):
        """
        Applies a white mask to an image based on a circular radius.

        Args:
            im (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with applied white mask.
        """
        im_size = np.shape(im)[0]
        g_loc = np.linspace(-10, 10, im_size)
        x, y = np.meshgrid(g_loc, g_loc)
        y = -y
        rad = np.abs(x + y * 1j)
        select = np.ones([im_size, im_size])
        select[rad > 10] = 0
        w_mask = np.zeros([im_size, im_size])
        w_mask[rad > 10] = np.nan
        im = im + w_mask
        return im

    def set_reponse_profiles(self):
        """
        Calculate and store response profiles for the active layer in the neural network model.

        This function calculates response profiles for each pixel (neuron) in the active layer's feature maps. The response
        profile for a neuron captures its activity across multiple stimuli. The calculated profiles are then stored in the
        corresponding dictionary (cnn_response_profile or rsl_response_profile) based on the active model type.

        This function assumes that the following attributes are properly set:
        - current_model_name: The name of the active model (either "cnn" or "rsl").
        - layer_name: The name of the active layer.
        - model_responses: A 3D array containing the responses of the model's feature maps to various stimuli.

        After execution, the calculated response profiles are stored in the appropriate dictionary entry.

        Note: The response profiles are stored as numpy arrays in the format:
        response_profiles[layer_name] = array(shape=(num_neurons, num_stimuli))

        Returns:
            None
        """
        print("Creating pixel vectors from the feature maps")
        response_profiles = []
        for x in range(self.n_out):
            for y in range(self.n_out):
                response_profiles.append(
                    np.asarray(self.model_responses[:, y, x]))

        if self.current_model_name == "cnn":
            self.cnn_response_profile[self.layer_name] = np.asarray(response_profiles)
            self.save_data(self.cnn_response_profile[self.layer_name], "model_responses")
        elif self.current_model_name == "rsl":
            self.rsl_response_profile[self.layer_name] = np.asarray(response_profiles)
            self.save_data(self.rsl_response_profile[self.layer_name], "model_responses")

    def set_gauss_response_profiles(self):
        """
        Saves Gaussian response profiles for different stimulus conditions.
        """
        with h5py.File("results/prf/gauss/gaussians_{}_{}.h5".format(self.locations, self.gaussians), 'w') as f:
            f.create_dataset("gaussian_activity", data=self.gauss_responses)
            f.create_dataset("gaussian_parameters", data=self.gaus_parameters)
            f.close()

    def get_gauss_response_data(self, locations=128, gaussians=20):
        """
        Retrieve Gaussian response profiles from a stored file.

        This function retrieves the previously generated Gaussian response profiles along with their associated Gaussian
        parameters from a stored file. The stored file contains information about how the model's feature maps respond to
        different Gaussian stimuli at specific locations and with different parameters.

        Parameters:
            locations (int): The number of different locations where Gaussian stimuli were presented.
            gaussians (int): The number of different Gaussian stimuli used in the analysis.

        After execution, the retrieved Gaussian response profiles and parameters are stored in the respective class attributes.

        Returns:
            None
        """
        try:
            locations = self.locations
            gaussians = self.gaussians
        except:
            None
        path = f"results/prf/gauss/gaussians_{locations}_{gaussians}.h5"
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                self.gauss_responses = f.get("gaussian_activity")[()]
                self.gaus_parameters = f.get("gaussian_parameters")[()]
                f.close()
        else:
            print("Gaussian response profiles do not exist...")
            sys.exit()

    def get_model_response_data(self):
        """
        Retrieve previously generated model response profiles from stored files.

        This function retrieves the previously generated model response profiles from stored files for different model layers.
        These response profiles capture the model's activation patterns in response to various stimuli and orientations.
        The stored files contain information about how the model's feature maps respond to different stimuli.

        This function assumes that the following attributes are properly set:
        - current_model_name: The name of the active model (either "cnn" or "rsl").
        - layers: A list of layer indices or names for which response profiles need to be retrieved.

        After execution, the retrieved model response profiles are stored in the active_response_profile dictionary.

        Returns:
            None
        """
        for l_name in self.layers:
            path = f"results/prf/{self.current_model_name}/model_responses_{l_name}.h5"
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    test = f.get("model_responses")
                    self.active_response_profile[l_name] = f.get("model_responses")[()]
                    f.close()
            else:
                print("Model response profiles do not exist...")
                sys.exit()

    def save_data(self, data, add):
        """
        Save the provided data into a file for future use.

        This function takes the provided data and saves it into a file with a specific filename pattern for later use.
        The saved file contains the provided data, which can be retrieved in subsequent analyses.

        Parameters:
            data: The data to be saved, such as response profiles.
            add (str): A descriptor to be included in the filename for identification.

        After execution, the provided data is saved into a file for later use.

        Returns:
            None
        """
        with h5py.File(f"results/prf/{self.current_model_name}/{add}_{self.layer_name}.h5", 'w') as f:
            f.create_dataset(add, data=data)
            f.close()

    def save_correlations(self):
        """
        Save calculated correlation data into a file for future reference.

        This function takes the calculated correlation data, such as radii, angles, sigmas, and correlation values,
        and saves them into a file with a specific filename pattern for future reference and analysis.

        After execution, the calculated correlation data is saved into a file for later use.

        Returns:
            None
        """
        with h5py.File(f"results/prf/{self.current_model_name}/correlation_data_{self.layer_name}.h5", 'w') as f:
            f.create_dataset("radius", data=self.radius)
            f.create_dataset("angle", data=self.angle)
            f.create_dataset("sigma", data=self.sigma)
            f.create_dataset("corr", data=self.gaus_corr)
            f.close()

    def load_correlations(self):
        """
        Load previously saved correlation data from a file.

        This function loads previously saved correlation data, including radii, angles, sigmas, and correlation values,
        from a stored file. The stored file contains information about the calculated correlations between the model's
        responses and Gaussian stimuli properties.

        After execution, the loaded correlation data is stored in the respective class attributes.

        Returns:
            None
        """
        with h5py.File(f"results/prf/{self.current_model_name}/correlation_data_{self.layer_name}.h5", 'r') as f:
            self.radius = f.get("radius")[()]
            self.angle = f.get("angle")[()]
            self.sigma = f.get("sigma")[()]
            self.gaus_corr = f.get("corr")[()]
            f.close()

    def load_stim(self, path, splitter, im_type="rgb", batch=None, vectorize=0, extend=0):
        """
        Load image stimuli from files.

        This function loads image stimuli from files located in a specified path. The loaded images can be either grayscale
        or RGB, based on the specified `im_type`. It supports loading images in batches or individual images.

        Parameters:
            path (str): The path to the folder containing the image stimuli.
            splitter (str): A character used to split filenames and extract image information.
            im_type (str): The type of images to load ("rgb" for RGB images, "gray" for grayscale images).
            batch (tuple or None): A tuple representing the range of image indices to load as a batch.
            vectorize (int): Indicates whether to vectorize the loaded images (1 for vectorization, 0 for no vectorization).
            extend (int): Indicates whether to extend the shape of the images (1 for extension, 0 for no extension).

        After execution, the loaded images are stored in the `batch` attribute.

        Returns:
            None
        """
        if im_type == "rgb":
            im_t = 1
        elif im_type == "gray":
            im_t = 0
        else:
            print("Incorrect Image Type")
            sys.exit()
        print("Loading images from: ", path)
        files = os.listdir(path)
        files = [f.split(splitter) for f in files]
        for i, e in enumerate(files):
            files[i][0] = int(files[i][0])
        files.sort()
        for i, e in enumerate(files):
            files[i][0] = str(files[i][0])
        files = [splitter.join(f) for f in files]
        if batch:
            files = files[int(batch[0]): int(batch[1])]
        images = np.asarray([cv2.imread(path + file, im_t) for file in files])
        # change 1 to 0 when corr
        print(f"Images loaded with shape:{images.shape}\n")

        if vectorize == 1:
            print("Vectorizing images...")
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])  # in corr, without ,3
            if extend == 1:
                images = images.reshape(images.shape[0], images.shape[1], 3)
        elif extend == 1:
            # images = images.reshape(images.shape[0], images.shape[1])
            images = images.reshape(images.shape[0], images.shape[1], images.shape[2])  # in corr, without ,3
        self.batch = np.asarray(images.astype('float32'))

    def show_featuremaps(self, maps, ver, hor):
        """
        Visualize a grid of feature maps.

        This function creates a grid of subplots, where each subplot displays a feature map from the provided array.
        The number of rows and columns in the grid is determined by the `ver` (vertical) and `hor` (horizontal) parameters.

        Parameters:
            maps (numpy.ndarray): The array containing the feature maps to be visualized.
            ver (int): The number of rows in the grid.
            hor (int): The number of columns in the grid.

        After execution, a visualization of the feature maps is displayed.

        Returns:
            None
        """
        ix = 1
        for _ in range(ver):
            for _ in range(hor):
                # specify subplot and turn of axis
                ax = plt.subplot(ver, hor, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(maps[ix - 1, :, :], cmap='jet')
                ix += 1
        # show the figure
        plt.colorbar()
        plt.show()

    def gen_gaus_parameters(self):
        """
        Generate parameters for Gaussian stimuli.

        This function generates parameters for Gaussian stimuli, including x-coordinate, y-coordinate, sigma (standard deviation),
        angle, and radial distance. These parameters define the properties of the Gaussian stimuli that will be used for analysis.

        After execution, the generated Gaussian parameters are stored in the `gaus_parameters` attribute.

        Returns:
            None
        """
        g_loc = np.linspace(-10, 10, self.locations)
        sigmas = np.linspace(self.gaus_min, self.gaus_max, self.gaussians)  # Testing
        print("Generating Gaussian parameters")
        self.gaus_parameters = np.asarray([(x, y, s, np.rad2deg(np.angle(x + y * 1j)), np.abs(x + y * 1j)) for x in g_loc
                                      for y in g_loc for s in sigmas])
        print("Gaussian parameters have a shape of: ", self.gaus_parameters.shape)

    def get_gauss_response_profiles(self, locations=128, gaussians=20, batches=64, g_min=0.0250, g_max=1.6):
        """
        Generate response profiles for Gaussian stimuli.

        This function calculates response profiles for Gaussian stimuli using the trained neural network models.
        It iterates over different bar orientations and positions, and for each combination, calculates the response
        of the neural network to the Gaussian stimuli.

        Parameters:
            locations (int): Number of locations for Gaussian stimuli.
            gaussians (int): Number of Gaussian stimuli.
            batches (int): Number of batches for processing.
            g_min (float): Minimum standard deviation for Gaussian stimuli.
            g_max (float): Maximum standard deviation for Gaussian stimuli.

        After execution, the generated response profiles for Gaussian stimuli are stored in the `gauss_responses` attribute.

        Returns:
            None
        """
        self.locations = locations
        self.gaussians = gaussians
        self.gaus_min = g_min
        self.gaus_max = g_max
        self.gaus_folder = "stimuli/gaussians/"
        self.bar_masks = "stimuli/barMasks/"
        sub_folders = ["0_0/", "1_90/", "2_45/", "3_135/"]
        self.gaus_num = self.locations ** 2 * self.gaussians
        self.batches = batches
        batch_size = self.gaus_num / self.batches

        self.gen_gaus_parameters()

        for i, folder in enumerate(sub_folders):
            if i == 0:
                self.load_stim(self.bar_masks + folder, '.', im_type="gray", vectorize=1, extend=0)
                bars = self.batch
            else:
                self.load_stim(self.bar_masks + folder, '.', im_type="gray", vectorize=1, extend=0)
                bars = np.append(bars, self.batch, axis=0)
        bars = bars / np.max(bars)
        self.gauss_responses = []

        for i in range(self.batches):
            print("Working on batch: {}, out of {}".format(i, batches))
            a = batch_size * i
            b = batch_size * (i + 1)
            self.load_stim(self.gaus_folder, '_', im_type="gray", batch=[a, b], vectorize=1, extend=0)
            self.gauss = self.batch
            self.gauss = self.gauss / np.max(self.gauss)
            self.gauss = self.gauss.T
            p = np.dot(bars, self.gauss) # This may be an issue!!! Check the .T
            print("Gaussian Activity profiles for this batch: {}\n".format(np.shape(p)))
            self.gauss_responses.append(p)
            print(np.shape(self.gauss_responses))

        self.gauss_responses = np.asarray(self.gauss_responses)
        self.gauss_responses = np.swapaxes(self.gauss_responses, 1, 2)
        self.gauss_responses = np.reshape(self.gauss_responses,
                                     (self.gauss_responses.shape[0] * self.gauss_responses.shape[1], self.gauss_responses.shape[2]))
        self.gauss_responses = self.gauss_responses.T
        self.set_gauss_response_profiles()

    def get_model_response_profiles(self, model, layers):
        """
        Obtain response profiles from trained neural network models.

        This function calculates response profiles from trained neural network models for different bar orientations and positions.
        It iterates over the specified layers and processes the response of the models for each combination of bar orientations
        and positions.

        Parameters:
            model (str): Name of the neural network model ("cnn" or "rsl").
            layers (list): List of layer indices to obtain response profiles from.

        After execution, the obtained response profiles are stored in the corresponding attributes (`cnn_response_profile` or `rsl_response_profile`).

        Returns:
            None
        """
        self.current_model_name = model
        for l_i, n_layer in enumerate(layers):
            self.layer_name = self.layers[n_layer]
            self.n_out = self.outsize[self.layer_name]
            if self.current_model_name == "cnn":
                print(f"Selecting CNN model")
                self.cut_model(n_layer)
                c_model = self.cnn_model_cut
                stim_folder = self.cnn_stim_folder
                response_profiles = self.cnn_response_profile
            elif self.current_model_name == "rsl":
                print(f"Selecting RSL model")
                self.cut_model(n_layer)
                c_model = self.rsl_model_cut
                stim_folder = self.rsl_stim_folder
                response_profiles = self.rsl_response_profile
            else:
                c_model = None
                print("Incorrect model name. Shutting down...")
                sys.exit()

            self.model_responses = np.empty([self.total_steps, self.n_out, self.n_out])
            c = 0
            print(f"Getting response profiles for layer {self.layer_name}")

            for i, folder in enumerate(self.sub_folders):
                if i in [0, 1]:
                    steps = self.steps_line
                else:
                    steps = self.steps_diagonal
                # Per bar orientation
                print(f"Working on orientation {i} out of {len(self.sub_folders)}...\n")
                for j in range(steps):
                    # Per bar location
                    print(f"Bar position {j} out of {steps}")
                    Average = np.zeros([self.n_out, self.n_out])
                    path = self.stim_root + stim_folder + folder + str(j) + "/"
                    self.load_stim(path, '_', vectorize=0, extend=0)
                    if np.max(self.batch) != 0.0:
                        self.batch = Normalize(self.batch, 7) # ---->>> WOW CHECK THIS! 7?
                    for it, s in enumerate(self.batch):
                        img = np.expand_dims(s[:, :, :], axis=0)
                        feature_maps = c_model.predict(img)
                        feature_maps = self.reshape_matrix(feature_maps)
                        #self.show_featuremaps(feature_maps, 8, 8)
                        # Get the average of the feature maps for 1 stimulus
                        activity = np.mean(feature_maps, axis=0)
                        #if show == 1:
                        #    show_image(activity)
                        # Update the average cell activities for that one bar location
                        Average = self.running_mean(Average, activity, it + 1)
                        # print(it)
                        # Max = np.max(Average, activity)
                    # We loop per bar orientation over all 12 positions.
                    # All feature maps are averaged over frequencies and offsets
                    # Addded to model_responses is the averaged activity map for all each
                    self.model_responses[c, :, :] = Average
                    c += 1

            self.model_responses = np.asarray(self.model_responses)
            print(f"The output shape of all the averaged feature maps for all orientations and {steps} positions: "
                  f"{self.model_responses.shape}")

            self.set_reponse_profiles()

    def get_correlations(self, model, layers):
        """
        Calculate correlations between model responses and Gaussian stimuli responses.

        This function calculates correlations between the response profiles obtained from trained neural network models
        and the response profiles of Gaussian stimuli. It iterates over the specified layers and calculates correlations
        for each combination of neural network responses and Gaussian stimuli responses.

        Parameters:
            model (str): Name of the neural network model ("cnn" or "rsl").
            layers (list): List of layer indices to calculate correlations for.

        After execution, the calculated correlations are stored in attributes like `radius`, `angle`, `sigma`, and `gaus_corr`.

        Returns:
            None
        """
        self.current_model_name = model
        if not self.gauss_responses:
            self.get_gauss_response_data()
        self.get_model_response_data()

        for ln in layers:
            self.layer_name = ln
            self.radius = np.empty([self.outsize[ln], self.outsize[ln]])
            self.angle = np.empty([self.outsize[ln], self.outsize[ln]])
            self.sigma = np.empty([self.outsize[ln], self.outsize[ln]])
            self.gaus_corr = np.empty([self.outsize[ln], self.outsize[ln]])

            print("Calculating Correlations and constructing results")
            i = 0

            for x in range(self.outsize[ln]):
                for y in range(self.outsize[ln]):
                    print("Working on pixel: ", i + 1, " out of ", self.outsize[ln] * self.outsize[ln])
                    print(np.mean(self.active_response_profile[ln][i, :]), np.std(self.active_response_profile[ln][i, :]))
                    pv = stats.zscore(self.active_response_profile[ln][i, :])
                    where_are_NaNs = np.isnan(pv)
                    pv[where_are_NaNs] = 0
                    # Gr = stats.zscore(Gauss_responses, axis=0)
                    # Gr = Gr.T
                    gr = stats.zscore(self.gauss_responses.T, axis=1)  # should be 1
                    r = np.asarray(self.correlation(pv, gr))
                    l = np.argmax(r)  # returning the index of the max value
                    # print('np.argmax(r): ',l)
                    # print('gausparameters: ',gaus_parameters[l])
                    specs = self.gaus_parameters[l]
                    self.radius[y, x] = specs[4]
                    self.angle[y, x] = specs[3]
                    self.sigma[y, x] = specs[2]
                    self.gaus_corr[y, x] = np.max(r)  # returning the correlation value
                    i += 1

            # Masking
            if model == "rsl":
                ret = RSL_File.RetinalCompression()
                self.radius = ret.generate_mask(self.radius)
                self.angle = ret.generate_mask(self.angle)
                self.sigma = ret.generate_mask(self.sigma)
                self.gaus_corr = ret.generate_mask(self.gaus_corr)

            self.save_correlations()

    def create_plots(self, model, layers):
        """
        Create plots based on calculated correlations and attributes.

        This function generates plots based on the calculated correlations and attributes like radius, angle, and sigma.
        It iterates over the specified layers and uses the stored correlation and parameter data to create visualizations.

        Parameters:
            model (str): Name of the neural network model ("cnn" or "rsl").
            layers (list): List of layer indices to create plots for.

        After execution, the generated plots are saved as SVG files in the `results/figures` directory.

        Returns:
            None
        """
        self.current_model_name = model

        for self.layer_name in layers:

            self.load_correlations()

            if model.lower() == "rsl":
                self.radius = self.white_mask(self.radius)
                self.angle = self.white_mask(self.angle)
                self.sigma = self.white_mask(self.sigma)
                self.gaus_corr = self.white_mask(self.gaus_corr)

            if self.layer_name == "V1":
                fig, ax = plt.subplots(figsize=(1.5, 1.0))
            elif self.layer_name == "V2":
                fig, ax = plt.subplots(figsize=(0.75, 0.5))
            else:
                fig, ax = plt.subplots(figsize=(0.375, 0.25))

            self.edit_figure(fig, ax, "sigma", self.sigma)
            self.edit_figure(fig, ax, "angle", self.angle)
            self.edit_figure(fig, ax, "radius", self.radius)

    def edit_figure(self, fig, ax, f_type, data):
        """
        Customize the appearance of the generated figures.

        This function customizes the appearance of the generated figures by modifying various visual elements.
        It sets up the colormap, adjusts color scaling, and handles the colorbar appearance based on the type of data.

        Parameters:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes._axes.Axes): The axes object.
            f_type (str): The type of figure being generated (e.g., "sigma", "angle", "radius").
            data (numpy.ndarray): The data array corresponding to the figure type.

        After execution, the customized figure is saved as an SVG file in the `results/figures` directory.

        Returns:
            None
        """
        matplotlib.rcParams['font.family'] = 'sans-serif'
        if f_type == "sigma":
            image = ax.imshow(data, cmap='jet', vmin=0, vmax=1.6)
        else:
            image = ax.imshow(data, cmap='jet')
        plt.rc('font', family='arial')
        plt.box(None)

        if self.layer_name == "V1":
            cb = plt.colorbar(image)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(7)
            tick_locator = matplotlib.ticker.MaxNLocator(nbins=2)
            cb.locator = tick_locator
            cb.update_ticks()
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(size=0)

        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'results/figures/{self.current_model_name}/{self.layer_name}_{f_type}.svg', dpi=600)
        try:
            cb.remove()
        except:
            None

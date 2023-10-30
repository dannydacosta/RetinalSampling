from tensorflow.python.keras.models import model_from_json
from ImSetNorm import Normalize
from scipy.stats import sem
import tensorflow.python.keras as keras
import pandas as pd
import glob
import cv2
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib


class AnalyzeEccSf:
    """
    A class for analyzing and visualizing data related to Eccentricity and Spatial Frequency using sinring stimuli.

    This class is designed to handle the analysis and visualization of neural network model outputs
    in the context of eccentricity and spatial frequency, with a focus on sinring stimuli. It includes methods
    to load pre-trained models, process sinring stimulus images, analyze specific model layers,
    and create customized plots to visualize the results.

    Attributes:
        root (str): Root directory path for various data and model folders.
        cnn_model_root (str): Sub-directory for CNN model.
        rsl_model_root (str): Sub-directory for RSL model.
        layer_numbers (dict): Mapping of layer names to their indices in the model's layer list.
        model (keras.Model): The loaded neural network model.
        model_cut (keras.Model): A sub-model cut at a specific layer.
        stim_root (str): Root directory path for stimulus images.
        cnn_stim_folder (str): Sub-directory for CNN stimulus images.
        rsl_stim_folder (str): Sub-directory for RSL stimulus images.
        sub_folders (list): Sub-folders for different eccentricities.
        layers (list): Names of neural network layers.
        outsize (dict): Mapping of layer names to their output sizes.
        rsl_result_folder (str): Result directory for RSL analysis.
        cnn_result_folder (str): Result directory for CNN analysis.
        savetype (str): File format for saving plots.
        frequency (list): List of spatial frequencies.
        cdeg_alt (list): List of circular degrees with alternative labels.
        cdeg (list): List of circular degrees.
        rads (list): List of radii corresponding to eccentricities.
        circ (list): List of circumferences corresponding to eccentricities.
        results (dict): Dictionary to store analysis results.
        results_keys (list): List of keys in the results dictionary.
        results_sem (dict): Dictionary to store analysis results' standard errors.

    Methods:
        __init__() -> None:
            Initialize class attributes with default values.

        load_model(model: str) -> None:
            Load a pre-trained neural network model based on the specified type ("cnn" or "rsl").

        select_model_layer(layer: str) -> None:
            Select a specific layer from the loaded model for further analysis.

        load_stim(path: str) -> numpy.ndarray:
            Load stimulus images from the specified path.

        g1(r, rad, ring) -> float:
            Calculate a Gaussian function value given parameters.

        reshape_matrix(matrix: numpy.ndarray) -> numpy.ndarray:
            Reshape a matrix for visualization.

        save_featuremaps(maps: numpy.ndarray, ecc: str, sf: str) -> None:
            Save feature maps as images for a specific eccentricity and spatial frequency.

        save_image(image: numpy.ndarray, name: str, root: str, ecc: str, sf: str) -> None:
            Save an image with a specific name and location.

        exp_decay_func(x, m, t, b) -> float:
            Exponential decay function for fitting.

        exp_func(x, t, b) -> float:
            Exponential function for fitting.

        f_test_regression(R2: float, p: float, n: int) -> Tuple[float, float]:
            Perform F-test for regression analysis.

        analyze_model(layers: list) -> None:
            Analyze the selected model layers using loaded stimuli.

        create_plot() -> None:
            Create a customized plot to visualize analyzed data.

    Note:
        - This class assumes that the required libraries (e.g., keras, numpy, pandas, matplotlib) are imported in the caller's scope.
        - Class attributes are initialized in the __init__() method.
        - Methods prefixed with @staticmethod are utility methods that don't rely on class instance attributes.
    """
    def __init__(self):
        self.root = "H:/RSLprojectLukas/RadialBias/"
        self.cnn_model_root = "CNN_model"
        self.rsl_model_root = "RSL_model"
        self.layer_numbers = {"V1": 0, "V2": 2, "v4":4}
        self.model = None
        self.model_cut = None

        self.stim_root = "stimuli/sinrings/"
        self.cnn_stim_folder = "undistorted/"
        self.rsl_stim_folder = "distorted/"
        self.sub_folders = ["1_Ecc/", "2.8_Ecc/", "4.7_Ecc/", "6.6_Ecc/", "8.5_Ecc/"]
        self.layers = ["V1", "V2", "V4"]
        self.outsize = {"V1": 128, "V2": 64, "V4": 32}
        self.rsl_result_folder = "results/eccSf/rsl/"
        self.cnn_result_folder = "results/eccSf/cnn/"
        self.savetype = 'pdf'

        self.frequency = [4, 10, 16, 22, 28, 34, 40, 46]
        self.cdeg_alt = ["0,", '2/π e', '3/π e', '4/π e', '5/π e', '6/π e', '7/π e', '8/π e', '9/π e', '10/π e']
        self.cdeg = ['2/π e', '5/π e', '8/π e', '11/π e', '14/π e', '17/π e', '20/π e', '23/π e']
        self.rads = [1, 2.8, 4.7, 6.6, 8.5]
        self.circ = [(2 * np.pi * r) for r in self.rads]

        self.results = {
            '1': [],
            '2.8': [],
            '4.7': [],
            '6.6': [],
            '8.5': [],
            "Spatial Frequency (cpr)": self.frequency
        }
        self.results_keys = list(self.results.keys())

        self.results_sem = {
            '1': [],
            '2.8': [],
            '4.7': [],
            '6.6': [],
            '8.5': [],
            "Spatial Frequency (cpr)": self.frequency
        }

    def load_model(self, model):
        """
                Load a pre-trained neural network model.

                Parameters:
                    model (str): The model type to load ("cnn" for CNN model, "rsl" for RSL model).

                Returns:
                    None
                """
        if model == "cnn":
            self.stim_folder = self.cnn_stim_folder
            self.model_folder = self.cnn_model_root
            self.result_folder = self.cnn_result_folder
        elif model == "rsl":
            self.stim_folder = self.rsl_stim_folder
            self.model_folder = self.rsl_model_root
            self.result_folder = self.rsl_result_folder
        # load json and create model
        json_file = open(self.model_folder + '/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.model_folder + "/model.h5")
        print("Loaded model from disk")
        print(self.model.summary())

    def select_model_layer(self, layer):
        """
            Select a specific layer from the loaded model.

            Parameters:
                layer (str): The layer name to select (e.g., "V1", "V2", "V4").

            Returns:
                None
        """
        layer_n = self.layer_numbers[layer]
        print(f"Cutting CNN model at layer {layer}")
        self.model_cut = keras.Model(inputs=self.model.input, outputs=self.model.layers[layer_n].output)

    @staticmethod
    def load_stim(path):
        """
            Load stimulus images from a given path.

            Parameters:
                path (str): The path to the folder containing stimulus images.

            Returns:
                numpy.ndarray: An array containing loaded stimulus images.
        """
        filepath = path + "*.jpg"
        files = glob.glob(filepath)
        images = [cv2.imread(file, 1) for file in files]
        images = np.array(images)
        return images

    #@staticmethod
    #def g1(r, rad, ring):
    #    return np.exp(-(np.square(r - rad) / (2 * np.square(ring))))

    def reshape_matrix(self, matrix):
        """
            Reshape a multi-dimensional matrix for visualization.

            This method reshapes a multi-dimensional input matrix into a two-dimensional array, where each column
            represents a single channel or feature. This reshaping is primarily used for visualization purposes.

            Parameters:
                matrix (numpy.ndarray): The input multi-dimensional matrix to be reshaped.

            Returns:
                numpy.ndarray: A two-dimensional array obtained by reshaping the input matrix.
        """
        output = np.asarray([matrix[0, :, :, i] for i in range(matrix.shape[3])])
        return output

    def save_featuremaps(self, maps, ecc, sf):
        """
            Save feature maps as images for a specific eccentricity and spatial frequency.

            This method takes a set of feature maps and saves them as image files for a particular eccentricity and
            spatial frequency combination. The images are saved in a designated result folder.

            Parameters:
                maps (numpy.ndarray): Feature maps to be saved as images.
                ecc (str): Eccentricity label for the saved images.
                sf (str): Spatial frequency label for the saved images.

            Returns:
                None
        """
        selection = [0, 1, 2, 5]
        maps = self.reshape_matrix(maps)
        os.makedirs(self.result_folder + ecc + sf, exist_ok=True)
        for i, e in enumerate(selection):
            m = maps[e, 2:-2, 2:-2]
            plt.imsave(self.result_folder + ecc + sf + "/{}.tiff".format(i), m)

    @staticmethod
    def save_image(image, name, root, ecc, sf):
        plt.imsave(root + name + "/" + ecc + sf + "/population.svg", image)

    @staticmethod
    def exp_decay_func(x, m, t, b):
        return m * np.exp(-t * x) + b

    @staticmethod
    def exp_func(x, t, b):
        return x ** t + b

    @staticmethod
    def f_test_regression(R2, p, n):
        dfn = p - 1
        dfd = n - dfn - 1
        numerator = R2 / dfn
        denominator = (1 - R2) / dfd
        f = numerator / denominator
        p = 1 - scipy.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
        return f, p

    def fontsizes(self, fontsize):  # region Fonts
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', family='arial')
        plt.rc('font', size=fontsize)  # controls default text size
        plt.rc('axes', titlesize=fontsize)  # fontsize of the title
        plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=fontsize)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=fontsize)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=fontsize)  # fontsize of the legend

    def analyze_model(self, layers):
        """
        Analyze the selected model layers using loaded sinring stimuli.

        This method performs analysis on the neural network model using the loaded sinring stimuli images.
        Prior to running this function, ensure that you have preloaded a model using the 'load_model()' method.

        For each specified layer, the method calculates the average response and standard error of the response
        across multiple loaded sinring stimuli images at different eccentricities. The analysis results are stored
        in the 'results' and 'results_sem' dictionaries of the class instance.

        Parameters:
            layers (list): A list of layer names to be analyzed.

        Returns:
            None

        Note:
            - Prior to running this function, you should use the 'load_model()' method to preload a neural network model.
            - The term "loaded sinring stimuli" refers to the set of stimulus images containing sinusoidal gratings
              arranged in ring-like patterns.
            - This method relies on the 'select_model_layer' method to choose the appropriate layer for analysis.
            - The analysis results are stored in the 'results' and 'results_sem' dictionaries,
              where each entry corresponds to a specific eccentricity.
        """
        self.values = []
        self.sem_values = []
        for layer in layers:
            self.select_model_layer(layer)

            for j, folder in enumerate(self.sub_folders):
                prediction = []
                sems = []
                path = self.stim_root + self.stim_folder + folder
                stim = self.load_stim(path)
                stim = np.asarray(Normalize(stim, 1))
                for k, img in enumerate(stim):
                    img = img.reshape(-1, 256, 256, 3)
                    feature_maps = self.model_cut.predict(img)
                    self.save_featuremaps(feature_maps, folder, self.cdeg[k])
                    activity = np.mean(feature_maps)
                    sem_l = sem(feature_maps, axis=None)
                    prediction.append(activity)
                    sems.append(sem_l)

                self.results[self.results_keys[j]] = np.divide(prediction, np.max(prediction))
                self.results_sem[self.results_keys[j]] = np.divide(sems, np.max(prediction))
                self.values.append(np.divide(prediction, np.max(prediction)))
                self.sem_values.append(np.divide(sems, np.max(prediction)))

    def create_plot(self):
        """
            Create a customized plot visualizing normalized layer responses.

            This method generates a customized plot to visualize the normalized layer responses based on the loaded
            sinring stimuli and the analyzed model outputs. The plot includes line plots representing different
            eccentricities and shaded regions indicating the standard error of the response.

            The x-axis of the plot corresponds to spatial frequency (in cycles per ring), while the y-axis represents
            the normalized layer response. The plot is designed to provide insights into how the neural network model's
            responses vary with different eccentricities and spatial frequencies.

            Parameters:
                None

            Returns:
                None

            Note:
                - Prior to running this function, ensure that you have loaded sinring stimuli and performed model analysis
                    using the 'load_stim' and 'analyze_model' methods.
                - The plot is customized with appropriate labels, colors, and styles to enhance readability.
                - The analysis results are used to generate the plot, including normalized responses and standard errors.
        """
        sf1_color = '#91C4FF'
        sf2_color = '#FFBF4A'
        sf3_color = '#76CC8C'
        sf4_color = '#FF7C7C'
        sf5_color = '#BCA3FF'

        line1_c = "#2c8eff"
        line2_c = "#ffa500"
        line3_c = "#34d15c"
        line4_c = "#ff3131"
        line5_c = "#7642ff"

        width = 1.0
        df = pd.DataFrame(self.results)
        sem = pd.DataFrame(self.results_sem)
        ms = 5

        self.fontsizes(7)
        fig1, ax = plt.subplots(figsize=(3.4, 2.4))
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', family='arial')
        ax.plot(self.frequency, self.values[0], marker=None, ms=ms, ls="-", linewidth=width, color=line1_c,
                label="1\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[0] - self.sem_values[0] * 1.96, self.values[0] + self.sem_values[0]
                        * 1.96, color=sf1_color, alpha=0.45)
        ax.plot(self.frequency, self.values[1], marker=None, ms=ms, ls="-", linewidth=width, color=line2_c,
                label="2.8\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[1] - self.sem_values[1] * 1.96, self.values[1] + self.sem_values[1]
                        * 1.96, color=sf2_color, alpha=0.45)
        ax.plot(self.frequency, self.values[2], marker=None, ms=ms, ls="-", linewidth=width, color=line3_c,
                label="4.7\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[2] - self.sem_values[2] * 1.96, self.values[2] + self.sem_values[2]
                        * 1.96, color=sf3_color, alpha=0.45)
        ax.plot(self.frequency, self.values[3], marker=None, ms=ms, ls="-", linewidth=width, color=line4_c,
                label="6.6\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[3] - self.sem_values[3] * 1.96, self.values[3] + self.sem_values[3]
                        * 1.96, color=sf4_color, alpha=0.45)
        ax.plot(self.frequency, self.values[4], marker=None, ms=ms, ls="-", linewidth=width, color=line5_c,
                label="8.5\N{DEGREE SIGN}")
        ax.fill_between(self.frequency, self.values[4] - self.sem_values[4] * 1.96, self.values[4] + self.sem_values[4]
                        * 1.96, color=sf5_color, alpha=0.45)
        x_labl = "Spatial Frequency (cpr)"
        plt.xlabel(x_labl, fontsize=7, family='sans-serif', labelpad=-2.5)
        plt.ylabel("Normalized Layer Response", fontsize=7, family='sans-serif', labelpad=-4)
        ax.set_xticks(self.frequency)
        ax.set_xticklabels(self.cdeg)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.minorticks_on()
        ax.tick_params(which='major', labelsize=7)
        ax.tick_params('both', length=00, width=0, which='minor', labelsize=7)
        plt.ylim(0.7, 1.01)
        ax.set_yticklabels(('0.7', '', '', '1.0'))
        plt.locator_params(axis="y", nbins=5)
        ax.set_xticklabels(('2/π e', '', '', '', '', '', '', '23/π e'))
        ax.legend(title=None, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), prop={'size': 7})
        plt.tight_layout()
        plt.savefig(self.result_folder + 'Normalized_Ecc' + '.' + self.savetype, dpi=600, format=self.savetype)
        plt.show()

    def fit_model(self):
        """
            Fit an exponential function to model preferences for eccentricity using curve fitting.

            This method calculates the best-fit exponential function to model preferences for eccentricity based on the
            normalized responses obtained from the analysis of the neural network model. The fitting is performed using
            the `scipy.optimize.curve_fit` function.

            The method first identifies the preferred spatial frequency for each eccentricity by finding the frequency
            with the highest normalized response. It then fits an exponential decay function to these preferred frequencies
            as a function of eccentricity.

            The fitted parameters, such as slope (m), time constant (t), and base level (b), are printed, along with
            the coefficient of determination (R²) and results of an F-test to assess the goodness of fit.

            Finally, a customized plot is generated to visualize the fitted exponential function alongside the actual
            model preferences. The plot includes appropriate labels, colors, and styles to enhance readability.

            Parameters:
                None

            Returns:
                None

            Note:
                - This method requires that the analysis results have been obtained through the 'analyze_model' method.
                - The fitting is performed using the 'exp_decay_func' defined in this class.
                - The fitted parameters and statistical values are printed for reference.
                - The plot generated showcases the fitted exponential function and model preferences.
            """
        model_pref = []
        for j in range(5):
            ind = self.results_keys[j]
            l = np.argmax(self.results[ind])
            freq = self.frequency[l]
            model_pref.append(freq)

        # perform the fit for exponential function
        p0 = (2000, .1, 50)  # start with values near those we expect
        params, cv = scipy.optimize.curve_fit(self.exp_decay_func, self.rads, model_pref, p0)
        m_Freq, t_Freq, b_Freq = params

        squaredDiffs = np.square(np.array(model_pref) - self.exp_decay_func(np.array(self.rads), m_Freq, t_Freq, b_Freq))
        squaredDiffsFromMean = np.square(np.array(model_pref) - np.mean(np.array(model_pref)))
        rSquared_exp_V1_Freq = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        sample_size = len(np.array(model_pref))
        f_exp_V1_Freq, p_exp_V1_Freq = self.f_test_regression(rSquared_exp_V1_Freq, 3, sample_size)
        print(f"m={m_Freq}, t={t_Freq}, b={b_Freq}, n={sample_size}")
        print(f"R² for for V1 RF exponential fitting = {rSquared_exp_V1_Freq} F = {f_exp_V1_Freq}, p = {p_exp_V1_Freq}")

        self.fontsizes(7)
        fig1, ax1 = plt.subplots(figsize=(2.8, 2.025))
        matplotlib.rcParams['font.family'] = 'sans-serif'
        plt.rc('font', family='arial')
        ax1.plot(self.rads, model_pref, marker=None, ls='-', linewidth=1.5, color='grey', label=self.model_folder)
        plt.plot(self.rads, self.exp_decay_func(np.array(self.rads), m_Freq, t_Freq, b_Freq), '--',
                                label="Fitted exponential function")
        ax1.legend(framealpha=1, markerscale=1, loc='upper right', frameon=False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_yticks(self.frequency)
        ax1.set_yticklabels(self.cdeg)
        ax1.set_xlim([1, 9])
        plt.locator_params(axis="x", nbins=9)
        ax1.set_yticklabels(('2/π e', '', '', '', '', '', '', '23/π e'), size=7)
        ax1.set_xticklabels(('1', '', '', '', '', '', '', '', '9'), size=7)
        ax1.minorticks_off()
        ax1.tick_params('both', length=00, width=0, which='minor', labelsize=7, size=7)
        plt.ylabel("cpr Preference", labelpad=-12, family='arial', fontsize=7)  # -7
        plt.xlabel("Eccentricity (degree)", labelpad=-2, family='arial', fontsize=7)
        plt.tight_layout()
        plt.savefig(self.result_folder + 'Ecc_degree' + '.' + self.savetype, dpi=600, format=self.savetype)
        plt.show()
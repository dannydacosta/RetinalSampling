import h5py
import numpy as np
import scipy
from scipy import stats as scipy_stats
import matplotlib
import matplotlib.pyplot as plt


class AnalyzePrf:
    """
    Initializes an instance of the AnalyzePrf class.

    This class performs advanced analyses on Population Receptive Field (pRF) mapping data acquired from previous mapping.
    It facilitates the loading of model-specific data, outlier removal, calculation of receptive field sizes, and
    generation of scatterplots for further insights.

    Parameters:
        None

    After initialization, the instance variables are:
    - layers (list): List of layer names ("V1", "V2", "V4").
    - model_data (dict): Dictionary to hold model-specific data dictionaries for each layer.
    - data_masks (dict): Dictionary to hold data masks for each layer.
    - outlier_masks (dict): Dictionary to hold outlier masks for each layer.

    Returns:
        None
    """
    def __init__(self):
        self.layers = ["V1", "V2", "V4"]
        self.model_data = {'V1': None, 'V2': None, 'V4': None}
        self.data_masks = {'V1': None, 'V2': None, 'V4': None}
        self.outlier_masks = {'V1': None, 'V2': None, 'V4': None}

    @staticmethod
    def lin_func(x, m, b):
        """
        Defines a linear function.

        This static method defines a linear function with parameters m (slope) and b (intercept) for further analysis.

        Parameters:
            x (numpy.ndarray): Input values.
            m (float): Slope of the linear function.
            b (float): Intercept of the linear function.

        Returns:
            numpy.ndarray: Output values of the linear function.
        """

        return m * x + b

    @staticmethod
    def exp_func(x, t, b):
        """
        Defines an exponential function.

        This static method defines an exponential function with parameters t (exponent) and b (offset) for further analysis.

        Parameters:
            x (numpy.ndarray): Input values.
            t (float): Exponent of the exponential function.
            b (float): Offset of the exponential function.

        Returns:
            numpy.ndarray: Output values of the exponential function.
        """
        return x ** t + b

    @staticmethod
    def f_test_regression(r2, p, n):
        """
        Performs an F-test for linear regression.

        This static method calculates the F-statistic and p-value for a linear regression model using the provided R-squared,
        number of predictors, and sample size for statistical analysis.

        Parameters:
            r2 (float): R-squared value of the linear regression.
            p (int): Number of predictors in the linear regression.
            n (int): Sample size.

        Returns:
            float: F-statistic.
            float: p-value of the F-test.
        """
        dfn = p - 1
        dfd = n - dfn - 1
        numerator = r2 / dfn
        denominator = (1 - r2) / dfd
        f = numerator / denominator
        p = 1 - scipy_stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
        return f, p

    def load_layer_data(self, model, layers=None):
        """
        Load pRF mapping data for the specified layers and model.

        This method loads pRF mapping data for specified layers and a given model. It populates the model_data dictionary
        with data related to eccentricity, angle, sigma, and correlation for each layer.

        Parameters:
            model (str): Name of the model.
            layers (list or None): List of layer names to load data for.

        Returns:
            None
        """
        self.current_model_name = model
        if layers:
            self.layers = layers
        for layer_name in self.layers:
            data_dict = {}
            with h5py.File(f"results/prf/{self.current_model_name}/correlation_data_{layer_name}.h5", 'r') as f:
                data_dict['eccentricity'] = np.nan_to_num(f.get("radius")[()])
                data_dict['angle'] = np.nan_to_num(f.get("angle")[()])
                data_dict['sigma'] = np.nan_to_num(f.get("sigma")[()])
                data_dict['correlation'] = np.nan_to_num(f.get("corr")[()])
                f.close()
            self.model_data[layer_name] = data_dict

    def create_pa_masks(self, data_dict):
        """
        Create data masks based on binned angles.

        This method creates data masks based on the binned angles of the pRF data. It segments data points into angle bins
        for further analysis.

        Parameters:
            data_dict (dict): Dictionary containing pRF data.

        Returns:
            dict: Dictionary containing data masks for angle bins.
        """
        bin_angle = 5
        self.degrees_bin0 = np.where((data_dict['angle'] <= 0 + bin_angle) & (data_dict['angle'] > 0))
        self.degrees_bin0_2 = np.where((data_dict['angle'] >= 0 - bin_angle) & (data_dict['angle'] < 0))
        self.degrees_bin90 = np.where((data_dict['angle'] >= 90 - bin_angle) & (data_dict['angle'] <= 90 + bin_angle))

        return data_dict

    def outlier_removal(self):
        """
        Identify and remove outliers from pRF data.

        This method identifies and removes outliers from the pRF data based on correlation values. It updates the
        outlier_masks dictionary with indices of data points that are considered outliers.

        Parameters:
            None

        Returns:
            None
        """
        for layer in self.layers:
            data_dict = self.model_data[layer]
            mask_dict = self.model_data[layer]

            correlations = data_dict['correlation']
            corr_mean = np.mean(correlations[correlations > 0])
            corr_std = np.std(correlations[correlations > 0])
            outlier_threshold = corr_mean - (3 * corr_std)
            self.outlier_masks[layer] = np.where(correlations > outlier_threshold)

    def rf_size(self):
        """
        Calculate and analyze receptive field sizes.

        This method calculates and analyzes receptive field sizes based on the pRF data. It performs linear fitting,
        calculates R-squared values, and performs F-tests for the calculated linear model for further analysis.

        Parameters:
            None

        Returns:
            None
        """
        for layer in self.layers:
            data_dict = self.model_data[layer]
            mask = self.outlier_masks[layer]

            sigma = data_dict['sigma']
            radius = data_dict['eccentricity']
            m, b = np.polyfit(np.ndarray.flatten(radius[mask]), np.ndarray.flatten(sigma[mask]), 1)
            x_value = np.linspace(radius.min(), radius.max(), 20000)

            squaredDiffs = np.square(np.ndarray.flatten(sigma[mask]) - self.lin_func(np.ndarray.flatten(radius[mask]), m, b))
            squaredDiffsFromMean = np.square(sigma[mask] - np.mean(sigma[mask]))
            rSquared_lin = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
            sample_size = len(sigma[mask])
            f_lin, p_lin = self.f_test_regression(rSquared_lin, 2, sample_size)
            print(f"R² for {self.current_model_name} {layer} RF size linear fitting = {rSquared_lin} F = {f_lin}, p = {p_lin}")

            self.scatterplot(layer, radius, sigma, x_value, m, b)
        plt.show()

    def scatterplot(self, layer, radius, sigma, x_value, m, b):
        """
        Generate scatterplots with fitted lines for receptive field sizes.

        This method generates scatterplots with fitted linear lines for analyzing receptive field sizes based on the pRF data.

        Parameters:
            layer (str): Layer name.
            radius (numpy.ndarray): Eccentricity data.
            sigma (numpy.ndarray): Sigma data.
            x_value (numpy.ndarray): X-axis values for the fitted line.
            m (float): Slope of the fitted line.
            b (float): Intercept of the fitted line.

        Returns:
            None
        """
        if layer == "V1":
            col = '#EAC152'
            matplotlib.rcParams['font.family'] = 'sans-serif'
            self.fig, self.ax = plt.subplots(figsize=(3.5, 2.5))
            plt.rc('font', family='arial')
        elif layer == "V2":
            col = '#91C4FF'
        else:
            col = '#76CC8C'

        # Create Scatterplot with fitted lines for RSL model
        self.ax.scatter(radius, sigma, s=0.3, c=col, label=f'{layer}')
        plt.plot(x_value, m * x_value + b, c='grey', linewidth=1.5)
        # V1_RSL_quadfit = plt.plot(x_value, exp_func(x_value, V1_t_s, V1_b_s), '--')
        plt.title(' ')
        self.ax.legend(markerscale=3, frameon=False, bbox_to_anchor=(0.2, 1), prop={'size': 7})
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        plt.text(5, -0.2, 'Eccentricity (deg)', fontsize=7, family='sans-serif', verticalalignment='center',
                 horizontalalignment='center')
        plt.text(-0.75, 1, 'Sigma (deg)', fontsize=7, family='sans-serif', verticalalignment='center',
                 horizontalalignment='center', rotation=90)
        plt.xlim([0, 10])
        plt.ylim(0, 2)
        plt.locator_params(axis="x", nbins=10)
        plt.locator_params(axis="y", nbins=5)
        self.ax.minorticks_on()
        self.ax.tick_params(which='major', labelsize=7)
        self.ax.tick_params('both', length=00, width=0, which='minor', labelsize=7)
        # ax.tick_params('y', length=1, width=1, which='minor', labelsize=7)
        self.ax.set_xticklabels(('', '', '', '', '', '', '', '', '', '', '10',))
        self.ax.set_yticklabels(('', '', '', '', '2'))
        # plt.text(10, -0.11, '10', fontsize=7, verticalalignment='center', horizontalalignment='center')
        plt.text(-0.38, -0.11, '0', fontsize=7, family='sans-serif', verticalalignment='center',
                 horizontalalignment='center')
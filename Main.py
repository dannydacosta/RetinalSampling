from Analyze_Retinotopy import PrfMapping
from Analyze_PRF import AnalyzePrf
from Analyze_EccSF import AnalyzeEccSf
from Analyze_RadialBias import RbRatioAnalyses


if __name__ == '__main__':
    # Create an instance of PrfMapping class
    prf = PrfMapping(fov=20)

    # Obtain model response profiles for specific conditions
    prf.get_model_response_profiles("cnn", [4])
    prf.get_model_response_profiles("rsl", [0])

    # Compute Gaussian response profiles
    prf.get_gauss_response_profiles()

    # Calculate correlations for specific layers and models
    prf.get_correlations("rsl", ["V1", "V2", "V4"])
    prf.get_correlations("cnn", ["V1", "V2", "V4"])

    # Create plots for model response profiles
    prf.create_plots("rsl", ["V1", "V2", "V4"])
    prf.create_plots("cnn", ["V1", "V2", "V4"])

    # Create an instance of AnalyzePrf class
    analyzePrfMapping = AnalyzePrf()

    # Load layer data and perform outlier removal for CNN model
    analyzePrfMapping.load_layer_data("cnn")
    analyzePrfMapping.outlier_removal()
    analyzePrfMapping.rf_size()

    # Load layer data and perform outlier removal for RSL model
    analyzePrfMapping.load_layer_data("rsl")
    analyzePrfMapping.outlier_removal()
    analyzePrfMapping.rf_size()

    # Create an instance of AnalyzeEccSf class
    eccSfExp = AnalyzeEccSf()

    # Load model and perform eccentricity and spatial frequency analysis for CNN model
    eccSfExp.load_model("cnn")
    eccSfExp.analyze_model(["V1"])
    eccSfExp.create_plot()
    eccSfExp.fit_model()

    # Load model and perform eccentricity and spatial frequency analysis for RSL model
    eccSfExp = AnalyzeEccSf()
    eccSfExp.load_model("rsl")
    eccSfExp.analyze_model(["V1"])
    eccSfExp.create_plot()
    eccSfExp.fit_model()

    # Create an instance of RbRatioAnalyses class
    rbExp = RbRatioAnalyses()

    # Compute model responses for radial bias analysis
    rbExp.model_responses("cnn")
    rbExp.model_responses("rsl")

    # Save model results for population and orientation maps
    rbExp.save_model_results("cnn")
    rbExp.save_model_results("rsl")

    # Create radial bins for bin activities calculation
    rbExp.create_radial_bins()

    # Calculate and store binned activities for CNN model
    rbExp.get_bin_activities("cnn")
    rbExp.get_bin_activities("rsl")

    # Save binned activities for CNN and RSL models
    rbExp.save_bin_activities("cnn")
    rbExp.save_bin_activities("rsl")

    # Compute radial bias ratios for CNN and RSL models
    rbExp.rb_ratio("cnn")
    rbExp.rb_ratio("rsl")

    # Create polar plots for radial bias analysis
    rbExp.create_polar_plots("cnn", [2, 5, 10, 20, 40, 60, 80, 100, 120])
    rbExp.create_polar_plots("rsl", [2, 5, 10, 20, 40, 60, 80, 100, 120])

    # Create and save radial bias ratio graphs for CNN and RSL models
    rbExp.rb_ratio_graph("cnn")
    rbExp.rb_ratio_graph("rsl")

    # Obtain and print radial bias ratios for specific conditions and models
    rbExp.get_ratios(4, [45.0, 135.0, 0.0, 90.0], "cnn")
    rbExp.get_ratios(4, [45.0, 135.0, 0.0, 90.0], "rsl")




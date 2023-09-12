import numpy as np
from scipy.stats import norm
from typing import List, Tuple
import matplotlib.pyplot as plt

class ABTestingAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def calculate_required_sample_size(effect_size: float, alpha: float = 0.05, power: float = 0.9, pooled_proportion: float = 0.5) -> int:
        """
        Calculate required sample size for a given effect size, significance level, and power.
        
        Args:
        - effect_size (float): Expected difference in proportions between treatment and control.
        - alpha (float): Significance level.
        - power (float): Desired power of the test.
        - pooled_proportion (float): Estimate of the proportion when the two groups are combined.
        
        Returns:
        - n (int): Required sample size per group.
        """
        
        # Z-scores for the given significance level and power
        z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed test
        z_beta = norm.ppf(power)
        
        # Pooled standard error
        pooled_se = np.sqrt(2 * pooled_proportion * (1 - pooled_proportion))
        
        # Required sample size per group
        n = ((z_alpha + z_beta) * pooled_se / effect_size) ** 2
        
        return int(np.ceil(n))  # Rounding up to ensure power is achieved

    @staticmethod
    def compute_success_and_sample_size(data: List[int]) -> Tuple[int, int]:
        """
        Compute the number of successes and the sample size from the given data.

        Args:
        - data (List[int]): Array containing binary values (0 or 1).

        Returns:
        - Tuple[int, int]: Number of successes and the sample size.
        """
        sample_size = len(data)
        success = sum(data)
        return success, sample_size

    @staticmethod
    def is_symmetric(success: int, sample_size: int, minimum_size: int) -> bool:
        """
        Check if the success distribution is symmetric based on the sample size and success count.

        Args:
        - success (int): Number of successes.
        - sample_size (int): Total sample size.
        - minimum_size (int): Minimum size for symmetry.

        Returns:
        - bool: True if symmetric, False otherwise.
        """
        p = success / sample_size
        return sample_size * p > minimum_size and sample_size * (1 - p) > minimum_size

    @staticmethod
    def symmetry_test(success_control: int, n_control: int, success_treatment: int, n_treatment: int, minimum_size: int = 10) -> str:
        control_symmetric = ABTestingAnalyzer.is_symmetric(success_control, n_control, minimum_size=minimum_size)
        treatment_symmetric = ABTestingAnalyzer.is_symmetric(success_treatment, n_treatment, minimum_size=minimum_size)

        if control_symmetric and treatment_symmetric:
            return "Both control and treatment distributions are symmetric enough."
        elif control_symmetric:
            return "Only the control distribution symmetric."
        elif treatment_symmetric:
            return "Only the treatment distribution is symmetric."
        else:
            return "Neither control nor treatment distributions are symmetric."

    @staticmethod
    def plot_distributions(ax: plt.Axes, control: List[int], treatment: List[int], title: str = "Distribution of 0s and 1s for Control and Treatment") -> None:
        """
        Plot distributions for control and treatment groups on the provided axis.

        Args:
        - ax (plt.Axes): Matplotlib axis to plot on.
        - control (List[int]): Array containing binary values for control group.
        - treatment (List[int]): Array containing binary values for treatment group.
        - title (str): Title for the plot.

        Returns:
        - None
        """
        # Define the bin edges
        bins = [-0.5, 0.5, 1.5]

        # Create histograms
        ax.hist(control, bins=bins, edgecolor="k", align="mid", alpha=0.5, label="Control")
        ax.hist(treatment, bins=bins, edgecolor="k", align="mid", alpha=0.5, label="Treatment")

        # Labeling and other aesthetics
        ax.set_xticks([0, 1])
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(loc="upper right")

    @staticmethod
    def describe_sampling_distribution(success: int, sample_size: int, alpha: float) -> Tuple[float, float, int, Tuple[float, float], float]:
        sample_mean = success / sample_size
        area_under_the_curve = 1-alpha/2

        sample_se = np.sqrt(sample_mean * (1 - sample_mean) / sample_size)

        critical_value = norm.ppf(area_under_the_curve)
        conf_int_lower = sample_mean - critical_value*sample_se
        conf_int_upper = sample_mean + critical_value*sample_se
            
        return sample_mean, sample_se, sample_size, (conf_int_lower, conf_int_upper), alpha
    
    @staticmethod
    def hypothesis_test_comparing_proportions(n_control: int, n_treatment: int, success_control: int, success_treatment: int, test_type: str, alpha: float) -> Tuple[float, bool, str, float, Tuple[float, float]]:
        p1 = success_control / n_control
        p2 = success_treatment / n_treatment

        odds_control = p1 / (1 - p1)
        odds_treatment = p2 / (1 - p2)
        
        odds_ratio = odds_treatment / odds_control

        pooled_p = (p1 * n_control + p2 * n_treatment) / (n_control + n_treatment)
        observed_diff = p2 - p1
        
        # Standard error using pooled proportion for hypothesis testing
        se_pooled = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
        
        # Compute the z-score for the observed difference using pooled SE
        z_score = observed_diff / se_pooled

        area_under_the_curve = norm.cdf(z_score)

        if test_type == 'right-tail':
            p_value = 1 - area_under_the_curve
        elif test_type == 'left-tail':
            p_value = area_under_the_curve
        elif test_type == 'two-tail':
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            raise ValueError("Invalid test type. Choose from ['right-tail', 'left-tail', 'two-tail']")

        # Calculate z_critical
        z_critical = norm.ppf(1 - alpha/2)

        # Standard error for the natural logarithm of the odds ratio
        se_ln_or = np.sqrt(1/success_control + 1/(n_control - success_control) + 1/success_treatment + 1/(n_treatment - success_treatment))
        
        # Confidence interval for the natural logarithm of the odds ratio
        ln_or_conf_int = (np.log(odds_ratio) - z_critical * se_ln_or, np.log(odds_ratio) + z_critical * se_ln_or)
        
        # Exponentiate to get the confidence interval for the odds ratio
        or_conf_int = (np.exp(ln_or_conf_int[0]), np.exp(ln_or_conf_int[1]))

        return p_value, p_value <= alpha, test_type, odds_ratio, or_conf_int

    @staticmethod
    def format_sampling_description(output: Tuple[float, float, int, Tuple[float, float], float], sample_type: str) -> str:
        """
        Format and describe the sampling distribution.

        Args:
        - output (tuple): Output from describe_sampling_distribution method.
        - sample_type (str): Type of sample, either 'control' or 'treatment'.

        Returns:
        - str: Formatted description of the sampling distribution.
        """
        if sample_type not in ['control', 'treatment']:
            return "Invalid sample type specified."

        sample_mean, sample_se, sample_size, conf_interval, alpha = output
        return (f"For the {sample_type} sample with a size of {sample_size}, the sample proportion is {sample_mean:.4f} "
                f"with a standard error of {sample_se:.4f}.\n"
                f"The {100*(1-alpha):.1f}% confidence interval for the sampling distribution of the sample proportion is "
                f"({conf_interval[0]:.4f}, {conf_interval[1]:.4f}).")

    @staticmethod
    def interpret_hypothesis_results(p_value: float, reject_null: bool, test_type: str, odds_ratio: float, conf_int: Tuple[float, float], alpha: float) -> str:
        """
        Provide an interpretation of the hypothesis test results.

        Args:
        - p_value (float): P-value from the hypothesis test.
        - reject_null (bool): Whether the null hypothesis is rejected.
        - test_type (str): Type of the test ('right-tail', 'left-tail', 'two-tail').
        - odds_ratio (float): Odds ratio value.
        - conf_int (tuple): Confidence interval for the odds ratio.
        - alpha (float): Significance level.

        Returns:
        - str: Interpretation of the hypothesis test results.
        """
        base_msg = f"Using a {test_type} hypothesis test with a significance level of {alpha:.4f} , "
        
        if reject_null:
            decision = "we reject the null hypothesis. "
            if test_type == 'right-tail':
                reason = "The treatment group has a significantly higher proportion of successes than the control group."
            elif test_type == 'left-tail':
                reason = "The treatment group has a significantly lower proportion of successes than the control group."
            elif test_type == 'two-tail':
                reason = "There's a significant difference in the proportions of successes between the treatment and control groups."
        else:
            decision = "we fail to reject the null hypothesis. "
            reason = "There's no significant evidence to suggest a difference in the proportions of successes between the two groups."
        
        # Interpreting the odds ratio
        if odds_ratio > 1:
            odds_message = f"Furthermore, the odds of success in the treatment group are approximately {odds_ratio:.2f} times the odds in the control group."
        elif odds_ratio == 1:
            odds_message = "The odds of success in the treatment group are the same as the odds in the control group."
        else:
            odds_message = f"Furthermore, the odds of success in the treatment group are {1/odds_ratio:.2f} times less than the odds in the control group."

        # Interpreting the confidence interval for the difference in proportions
        conf_int_message = (f"The {100*(1-alpha):.2f} confidence interval for the odds ratio between treatment and control groups is "
                            f"({conf_int[0]:.4f}, {conf_int[1]:.4f}).")

        return base_msg + decision + reason + "\n" + odds_message + "\n" + conf_int_message

    @staticmethod
    def plot_odds_ratio_with_ci(odds_ratio: float, odds_ratio_ci: Tuple[float, float]) -> None:
        """
        Plot the odds ratio with its confidence interval.

        Parameters:
        - odds_ratio (float): The calculated odds ratio.
        - odds_ratio_ci (tuple): The confidence interval of the odds ratio.
        """

        # Calculating the error distances for both sides of the odds ratio
        lower_error = odds_ratio - odds_ratio_ci[0]
        upper_error = odds_ratio_ci[1] - odds_ratio

        # Plotting the odds-ratio
        plt.figure(figsize=(10, 4))
        plt.errorbar(x=[odds_ratio], y=[0], xerr=[[lower_error], [upper_error]], fmt='o', color='blue', capsize=5)
        plt.xscale('log')
        plt.yticks([])
        plt.axvline(1, color='grey', linestyle='--')
        plt.title("Odds Ratio with Confidence Interval")
        plt.show()
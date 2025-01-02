import numpy as np
import gc

# costs pulled from https://app.powerbi.com/view?r=eyJrIjoiZGQxYjNkOGUtOTIwMC00N2VjLWEyM2EtYjAzOGMwNWU5ODQ1IiwidCI6IjM3YzM1NGIyLTg1YjAtNDdmNS1iMjIyLTA3YjQ4ZDc3NGVlMyJ9
# surgery costs are in 'admitted patient care' section of NHS reference costs

# age standardised rate of death - https://www.ssa.gov/oact/STATS/table4c6.html#fn1
mort_other_dict = {0: 0.00586, 1: 0.00042, 2: 0.000272, 3: 0.000225, 4: 0.000184, 5: 0.000157, 6: 0.00014, 7: 0.000128, 8: 0.000122, 9: 0.000123, 10: 0.000129, 11: 0.000138, 12: 0.000164, 13: 0.00022, 14: 0.00031, 15: 0.000446, 16: 0.000637, 17: 0.000868, 18: 0.0011, 19: 0.00127, 20: 0.001373, 21: 0.001488, 22: 0.001605, 23: 0.001714, 24: 0.001835, 25: 0.001963, 26: 0.002082, 27: 0.002202, 28: 0.00233, 29: 0.002457, 30: 0.002574, 31: 0.002683, 32: 0.002787, 33: 0.002881, 34: 0.002974, 35: 0.003074, 36: 0.003175, 37: 0.003295, 38: 0.003444, 39: 0.003608, 40: 0.00378, 41: 0.003958, 42: 0.004144, 43: 0.004337, 44: 0.00454, 45: 0.004774, 46: 0.005064, 47: 0.005399, 48: 0.005796, 49: 0.006214, 50: 0.006671, 51: 0.007167, 52: 0.007736, 53: 0.008351, 54: 0.009035,
                        55: 0.00977, 56: 0.010567, 57: 0.011398, 58: 0.012291, 59: 0.013224, 60: 0.014267, 61: 0.015353, 62: 0.016484, 63: 0.017617, 64: 0.018759, 65: 0.019914, 66: 0.021104, 67: 0.022423, 68: 0.023847, 69: 0.025357, 70: 0.02705, 71: 0.02897, 72: 0.031188, 73: 0.033754, 74: 0.036747, 75: 0.040563, 76: 0.044308, 77: 0.048498, 78: 0.053229, 79: 0.058778, 80: 0.064617, 81: 0.070947, 82: 0.077834, 83: 0.085686, 84: 0.094809, 85: 0.10509, 86: 0.116592, 87: 0.129306, 88: 0.142732, 89: 0.157638, 90: 0.174458, 91: 0.193027, 92: 0.21293, 93: 0.232657, 94: 0.251826, 95: 0.270943, 96: 0.289756, 97: 0.307998, 98: 0.325393, 99: 0.341662, 100: 0.358746, 101: 0.376683, 102: 0.395517, 103: 0.415293, 104: 0.436058, 105: 0.45786, 106: 0.480753, 107: 0.504791, 108: 0.530031, 109: 0.556532,
                        110: 0.584359, 111: 0.613577, 112: 0.644256, 113: 0.676468, 114: 0.710292, 115: 0.745806, 116: 0.783097, 117: 0.822251, 118: 0.863364, 119: 0.906532}
#high level params to make sense and sanity check:
#   national cost of kidney cancer is about £6.5B per year: https://www.kidneyresearchuk.org/wp-content/uploads/2023/06/Economics-of-Kidney-Disease-summary-report_accessible.pdf
#   12-14k new cases per year, 4-5k deaths per year: https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/kidney-cancer

#This script simulates 8m people over 10 years, with a 2 year screening interval

### TODO

# - Track 5-year survival of cancers (benign, malignant)
# - Track life expectancy

# -----------------------
# Base Global Parameters
# -----------------------
BASE_PARAMS = {
    # Population and simulation (no variation)
    "N": 100_000, # 20 million between 45-74
    "years": 45,
    "screen_interval": 2,
    "yearly_print": False,

    # Disease model parameters (no variation)
    "annual_incidence": 0.0012,
    "initial_prevalence": 0.006,
    "pct_large": 0.57,
    "mean_size": 3.5,
    "std_size": 0.5,

    # Tumor size thresholds and malignancy (no variation)
    "very_small_threshold": 2.0,
    "large_threshold": 3.0,
    "threshold_2cm": 2.0,
    "threshold_4cm": 4.0,
    "prob_malignant_at_2cm": 0.25,
    "prob_malignant_at_4cm": 0.75,

    # Growth parameters
    "doubling_time_mean": 1.5,
    "doubling_time_sd": 0.5,

    # Costs (allowed to vary)
    "cost_benign_cure": 20000, #complex general abdominal procedures between 2-20k.
    # Complex, Open or Laparoscopic, Kidney or Ureter Procedures ranges between 3-18k
    # if intermediate, cost drops to 1-10k
    # Kidney, Urinary Tract or Prostate Neoplasms ranges between 300 - 20000
    # major robotic urological procedures are roughly 9k
    # Major, Open or Laparoscopic, Kidney or Ureter Procedures - between 900 - 20000
    # this roughly matches private costs: https://www.kingedwardvii.co.uk/services/radical-nephrectomy
    # also matches historic costs roughly: https://pubmed.ncbi.nlm.nih.gov/12474534/
    "cost_malignant_annual": 80000,
    # https://www.nice.org.uk/guidance/ta178/documents/nice-issues-guidance-on-the-use-of-other-treatment-options-for-renal-cancer
    # presumed incidental finding - no cost due to disease, scan for other reasons
    "cost_screen_ai": 10,
    "cost_screen_no_ai": 50,
    "cost_followup_ai": 25,
    "cost_followup_no_ai": 200,
    "cost_MDT": 220,
    #CDC CT scan with contrsat, over 19 - £110 in total. PResume £50 is for the overhead, £50 for the radiologist, £10 is contrast use
    # outpatients CT scan - presume £150 for the scan, £50 for the radiologist - 200

    # Annual detection probabilities (allowed to vary)
    "annual_detection_prob_benign": 0.1,
    "annual_detection_prob_malignant": 0.5,

    # Cure time clamp values (no variation) - cure in 0-2 years, if not dead
    "CURE_TIME_MIN": 1,
    "CURE_TIME_MAX": 3,

    # Mortality every year, copy the values from above and 1-value for each year
    "mort_other_dict": {k: 1-v for k,v in mort_other_dict.items()},
    "mort_benign": 0.96, #https://digital.nhs.uk/data-and-information/publications/statistical/cancer-survival-in-england/cancers-diagnosed-2016-to-2020-followed-up-to-2021
    "mort_malignant": 0.4,
    "starting_age_min": 45,
    "starting_age_max": 74,
    "stop_screening_age": 75,
    "start_screening_age": 55,

    # Initial tumour size for new cancers (allowed to vary)
    "INITIAL_TUMOUR_SIZE": 0.5,

    # Screening detection parameters (no variation)
    "det_params_no_screen": {
        'det_prob_very_small': 0.0,
        'det_prob_small': 0.0,
        'det_prob_large': 0.0,
        'specificity': 1.0,
        'onsite_followup':False
    },
    "det_params_screen_no_ai": {
        'det_prob_very_small': 0.5,
        'det_prob_small': 0.65,
        'det_prob_large': 0.97,
        'specificity': 0.952,
        'onsite_followup':False
    },
    "det_params_screen_ai_2stage": {
        'det_prob_very_small': 0.3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.97,
        'specificity': 0.808,
        'onsite_followup':True
    },
    "det_params_screen_ai_1stage": {
        'det_prob_very_small': 0.3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.97,
        'specificity': 0.808,
        'onsite_followup':False
    }
}

# Parameters not to vary
NO_VARIATION_KEYS = {
    "N","years","screen_interval","initial_prevalence","pct_large", "doubling_time_mean","doubling_time_sd",
    "mort_benign","mort_malignant","mort_other_dict","starting_age","under55mort","over100mort",
    "starting_age_min","starting_age_max", "stop_screening_age","start_screening_age",
    "very_small_threshold","large_threshold","threshold_2cm","threshold_4cm",
    "CURE_TIME_MIN","CURE_TIME_MAX","average_lifetime_malignant_check_cost"
    "det_params_no_screen","det_params_screen_no_ai","det_params_screen_ai"
}

def vary_parameter(value):
    """Vary the given parameter by ±20% and clamp if necessary."""
    factor = np.random.uniform(0.8, 1.2)
    new_val = value * factor
    # If value was a probability (0<=val<=1), clamp to [0,1]
    if isinstance(value, float):
        if 0 <= value <= 1:
            new_val = max(0.0, min(1.0, new_val))
        else:
            # For costs, mortality, sizes etc., just ensure positive
            new_val = max(1e-12, new_val)
    return new_val

def vary_all_params(base_params):
    """Create a perturbed version of BASE_PARAMS by ±20% variation for allowed parameters."""
    p = {}
    for k,v in base_params.items():
        if k in NO_VARIATION_KEYS:
            # No variation
            p[k] = v
        else:
            if isinstance(v, dict):
                # This should be detection params dictionary which we do not vary
                # But it's already covered by NO_VARIATION_KEYS, so theoretically won't get here
                p[k] = v
            else:
                if isinstance(v, (int, float)):
                    # Vary these parameters
                    new_v = vary_parameter(float(v))
                    p[k] = new_v
                else:
                    # Non-variable, just copy
                    p[k] = v
    return p

def run_simulation(params, screening_on, screening_AI,ai_stages=2, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if not screening_on:
        det = params["det_params_no_screen"]
    elif screening_AI and ai_stages == 2:
        det = params["det_params_screen_ai_2stage"]
    elif screening_AI and ai_stages == 1:
        det = params["det_params_screen_ai_1stage"]
    else:
        det = params["det_params_screen_no_ai"]

    det_prob_very_small = det['det_prob_very_small']
    det_prob_small = det['det_prob_small']
    det_prob_large = det['det_prob_large']
    specificity = det['specificity']
    onsite_followup = det['onsite_followup']
    yearly_print = params["yearly_print"]

    N = params["N"]
    years = params["years"]
    screen_interval = params["screen_interval"]
    starting_age_min = params["starting_age_min"]
    starting_age_max = params["starting_age_max"]
    stop_screening_age = params["stop_screening_age"] # when age>75, remove from screening
    start_screening_age = params["start_screening_age"] # when age>55, add to screening
    mort_other_dict = params["mort_other_dict"]

    # ages is an a uniformly distributed array of ages between starting_age_min and starting_age_max
    ages = np.random.randint(starting_age_min, starting_age_max+1, N)

    annual_incidence = params["annual_incidence"]
    initial_prevalence = params["initial_prevalence"]
    pct_large = params["pct_large"]
    mean_size = params["mean_size"]
    std_size = params["std_size"]

    very_small_threshold = params["very_small_threshold"]
    large_threshold = params["large_threshold"]
    threshold_2cm = params["threshold_2cm"]
    threshold_4cm = params["threshold_4cm"]
    prob_malignant_at_2cm = params["prob_malignant_at_2cm"]
    prob_malignant_at_4cm = params["prob_malignant_at_4cm"]

    doubling_time_mean = params["doubling_time_mean"]
    doubling_time_sd = params["doubling_time_sd"]

    mort_benign = params["mort_benign"]
    mort_malignant = params["mort_malignant"]


    cost_benign_cure = params["cost_benign_cure"]
    cost_malignant_annual = params["cost_malignant_annual"]
    cost_screen_ai = params["cost_screen_ai"]
    cost_screen_no_ai = params["cost_screen_no_ai"]
    cost_followup_ai = params["cost_followup_ai"]
    cost_followup_no_ai = params["cost_followup_no_ai"]
    cost_MDT = params["cost_MDT"]

    annual_detection_prob_benign = params["annual_detection_prob_benign"]
    annual_detection_prob_malignant = params["annual_detection_prob_malignant"]

    CURE_TIME_MIN = params["CURE_TIME_MIN"]
    CURE_TIME_MAX = params["CURE_TIME_MAX"]
    INITIAL_TUMOUR_SIZE = params["INITIAL_TUMOUR_SIZE"]

    screen_years = set(range(0, years+1, screen_interval))

    in_program = np.ones(N, dtype=bool)

    has_cancer = np.zeros(N, dtype=bool)
    tumor_size = np.zeros(N, dtype=float)
    detected = np.zeros(N, dtype=bool)
    malignant = np.zeros(N, dtype=bool)
    tumour_start_time = np.full(N, np.nan)
    tumour_growth_factor = np.ones(N)
    cure_year = np.full(N, np.nan)
    #populate mortality status using dict of mortalities and ages
    # for i in range(N):
    #     mortality_status[i] = mort_other_dict[ages[i]]

    #replace for-loop above with vectorized version
    mortality_status = np.vectorize(mort_other_dict.get)(ages)

    under_cure_malignant = np.zeros(N, dtype=bool)
    under_cure_benign = np.zeros(N, dtype=bool)

    sojourn_times = []

    false_positives = 0
    true_positives = 0
    followups = 0

    cost_screening_total = 0.0
    cost_followup_total = 0.0
    cost_detection_noscreen_total = 0.0
    cost_treatment_total = 0.0

    malignant_costs = 0.0
    benign_costs = 0.0

    #zero all counts
    benign_total_count = 0
    malignant_total_count = 0

    life_years_during_program = 0.0

    total_malignant_detected = 0
    total_benign_detected = 0

    def assign_exponential_growth(indices):
        if len(indices) == 0:
            return
        dt = np.random.normal(doubling_time_mean, doubling_time_sd, size=len(indices))
        dt = np.where(dt <= 0.1, 0.1, dt)
        growth_factors = 2 ** (1.0 / dt)
        tumour_growth_factor[indices] = growth_factors

    def update_malignancy_on_size_crossing(prev_sizes, new_sizes, idx):
        not_malig = ~malignant[idx]
        if np.any(not_malig):
            idx_nm = idx[not_malig]
            prev = prev_sizes[not_malig]
            new = new_sizes[not_malig]

            crossed_2cm = (prev <= threshold_2cm) & (new > threshold_2cm)
            if np.any(crossed_2cm):
                c2_indices = idx_nm[crossed_2cm]
                turn_malig_2 = np.random.rand(len(c2_indices)) < prob_malignant_at_2cm
                malignant[c2_indices[turn_malig_2]] = True

            still_not_malig = idx_nm[~malignant[idx_nm]]
            if len(still_not_malig) > 0:
                still_prev = prev[~malignant[idx_nm]]
                still_new = new[~malignant[idx_nm]]
                crossed_4cm = (still_prev <= threshold_4cm) & (still_new > threshold_4cm)
                if np.any(crossed_4cm):
                    c4_indices = still_not_malig[crossed_4cm]
                    turn_malig_4 = np.random.rand(len(c4_indices)) < prob_malignant_at_4cm
                    malignant[c4_indices[turn_malig_4]] = True

    # Initial prevalence - only for 55 and above
    initial_cancer_indices = np.where((ages >= start_screening_age) & (np.random.rand(N) < initial_prevalence))[0]
    num_initial_cancers = len(initial_cancer_indices)
    if num_initial_cancers > 0:
        initial_sizes = np.random.normal(loc=mean_size, scale=std_size, size=num_initial_cancers)
        sorted_indices = np.argsort(initial_sizes)
        num_large = int(np.round(pct_large * num_initial_cancers))
        large_indices = sorted_indices[-num_large:]
        small_indices = sorted_indices[:-num_large]

        tumor_size[initial_cancer_indices[large_indices]] = initial_sizes[large_indices]
        tumor_size[initial_cancer_indices[small_indices]] = initial_sizes[small_indices]
        has_cancer[initial_cancer_indices] = True

        assign_exponential_growth(initial_cancer_indices)

        init_new = tumor_size[initial_cancer_indices]
        init_idx = initial_cancer_indices
        above_4 = init_new > threshold_4cm
        if np.any(above_4):
            c4_indices = init_idx[above_4]
            turn_malig_4 = np.random.rand(len(c4_indices)) < prob_malignant_at_4cm
            malignant[c4_indices[turn_malig_4]] = True

        still_not_malig = init_idx[~malignant[init_idx]]
        if len(still_not_malig) > 0:
            sub_sizes = init_new[~malignant[init_idx]]
            above_2_only = (sub_sizes > threshold_2cm) & (sub_sizes <= threshold_4cm)
            if np.any(above_2_only):
                c2_indices = still_not_malig[above_2_only]
                turn_malig_2 = np.random.rand(len(c2_indices)) < prob_malignant_at_2cm
                malignant[c2_indices[turn_malig_2]] = True

    deaths_other_by_year = []
    deaths_benign_by_year = []
    deaths_malignant_by_year = []

    for year in range(0, years + 1):
        # Define program_indices at the start of each year
        program_indices = np.where(in_program)[0]

        # Cure check
        detected_indices = np.where(detected & in_program)[0]
        if len(detected_indices) > 0:
            has_cure_year = ~np.isnan(cure_year[detected_indices])
            if np.any(has_cure_year):
                candidates = detected_indices[has_cure_year]
                reached_cure = cure_year[candidates] <= year
                if np.any(reached_cure):
                    cured_indices = candidates[reached_cure]
                    benign_cured_mask = under_cure_benign[cured_indices]
                    benign_cured_count = np.sum(benign_cured_mask)
                    cost_treatment_total += benign_cured_count * cost_benign_cure
                    benign_costs += benign_cured_count * cost_benign_cure

                    #update for cured malignant
                    malignant_cured_mask = under_cure_malignant[cured_indices]
                    malignant_cured_count = np.sum(malignant_cured_mask)
                    cost_treatment_total += malignant_cured_count * (cost_benign_cure )
                    malignant_costs += malignant_cured_count * (cost_benign_cure )
                    #update mortality status for cured malignant based on age

                    mortality_status[cured_indices] = np.vectorize(mort_other_dict.get)(ages[cured_indices])
                    under_cure_malignant[cured_indices] = False
                    under_cure_benign[cured_indices] = False
                    malignant[cured_indices] = False
                    has_cancer[cured_indices] = False
                    detected[cured_indices] = False
                    tumor_size[cured_indices] = 0.0
                    tumour_growth_factor[cured_indices] = 1.0
                    cure_year[cured_indices] = np.nan

        # Malignant annual cost
        currently_malignant = malignant[in_program]
        currently_detected = detected[in_program]  # find all cancers, malignant and detected
        malignant_prevdetected = currently_detected & currently_malignant
        malignant_prevdetected_count = np.sum(malignant_prevdetected)
        cost_treatment_total += malignant_prevdetected_count * cost_malignant_annual
        malignant_costs += malignant_prevdetected_count * cost_malignant_annual

        malignant_newdetected_count = 0
        benign_newdetected_count = 0

        # Growth
        cancer_indices = np.where(in_program & has_cancer)[0]  # Corrected
        if len(cancer_indices) > 0:
            prev_sizes = tumor_size[cancer_indices].copy()
            tumor_size[cancer_indices] *= tumour_growth_factor[cancer_indices]
            new_sizes = tumor_size[cancer_indices]
            update_malignancy_on_size_crossing(prev_sizes, new_sizes, cancer_indices)  # Pass integer indices

        # Screening detection
        followups_this_screen = 0
        if screening_on and year in screen_years:
            program_indices = np.where(in_program)[0]  # Define program_indices correctly
            # check if the person is still in the program
            in_screening = in_program & (ages <= stop_screening_age) & (ages >= start_screening_age)
            alive_in_program_count = len(program_indices)
            if screening_AI:
                cost_screening_total += len(in_screening) * cost_screen_ai
            else:
                cost_screening_total += len(in_screening) * cost_screen_no_ai

            total_cancers_at_screen = len(cancer_indices)
            if total_cancers_at_screen > 0:
                sizes_current = tumor_size[cancer_indices]
                very_small_current = sizes_current <= very_small_threshold
                large_current = sizes_current > large_threshold
                small_current = ~(very_small_current | large_current)

                detect_very_small = np.random.rand(np.sum(very_small_current)) < det_prob_very_small
                detect_small = np.random.rand(np.sum(small_current)) < det_prob_small
                detect_large = np.random.rand(np.sum(large_current)) < det_prob_large

                detection_array = np.zeros(len(cancer_indices), dtype=bool)  # Corrected
                detection_array[very_small_current] = detect_very_small
                detection_array[small_current] = detect_small
                detection_array[large_current] = detect_large

                number_detected_this_screen = np.sum(detection_array)
                detected_cancer_indices = cancer_indices[detection_array]

                detected_sojourn = (year - tumour_start_time[detected_cancer_indices])
                sojourn_times.extend(detected_sojourn)

                malignant_detected_count = np.sum(malignant[detected_cancer_indices])

                benign_newdetected_count+=np.sum(~malignant[detected_cancer_indices])
                benign_total_count+=np.sum(~malignant[detected_cancer_indices])
                malignant_newdetected_count+=malignant_detected_count
                malignant_total_count+=malignant_detected_count

                true_positives += number_detected_this_screen

                in_screening_nocancer = in_screening[~has_cancer]
                false_detections = np.random.rand(len(in_screening_nocancer)) < (1 - specificity)
                false_positives += np.sum(false_detections)

                # Calculate follow-ups for this screening event only
                followups_this_screen = number_detected_this_screen + np.sum(false_detections)

                detected[detected_cancer_indices] = True
                tumor_size[detected_cancer_indices] = 0.0
                tumour_growth_factor[detected_cancer_indices] = 1.0
                tumour_start_time[detected_cancer_indices] = np.nan

                random_cure_offset = np.random.uniform(CURE_TIME_MIN, CURE_TIME_MAX, size=len(detected_cancer_indices))
                cure_year[detected_cancer_indices] = year + 1 + random_cure_offset

                malignant_at_detection = malignant[detected_cancer_indices]

                under_cure_malignant[detected_cancer_indices[malignant_at_detection]] = True
                under_cure_benign[detected_cancer_indices[~malignant_at_detection]] = True
            else:
                followups_this_screen = 0
                number_detected_this_screen = 0

            # Follow-up cost
            if followups_this_screen > 0:
                followups += followups_this_screen
                if onsite_followup:
                    cost_followup_total += followups_this_screen * cost_followup_ai
                else:
                    cost_followup_total += followups_this_screen * cost_followup_no_ai

            # Add MDT cost to all detected tumors
            cost_followup_total += number_detected_this_screen * cost_MDT
            benign_costs += benign_newdetected_count * cost_MDT
            malignant_costs += malignant_newdetected_count * cost_MDT

        # Annual detection for all undetected cancers
        undetected_cancer_indices = np.where(in_program & ~detected & has_cancer)[0]
        if len(undetected_cancer_indices) > 0:
            malignant_undetected_indices = undetected_cancer_indices[malignant[undetected_cancer_indices]]
            benign_undetected_indices = undetected_cancer_indices[~malignant[undetected_cancer_indices]]

            detected_benign_indices = benign_undetected_indices[np.random.rand(len(benign_undetected_indices)) < annual_detection_prob_benign] if len(benign_undetected_indices) > 0 else np.array([], dtype=int)
            detected_malignant_indices = malignant_undetected_indices[np.random.rand(len(malignant_undetected_indices)) < annual_detection_prob_malignant] if len(malignant_undetected_indices) > 0 else np.array([], dtype=int)

            detected_annual_indices = np.concatenate((detected_benign_indices, detected_malignant_indices))
            if len(detected_annual_indices) > 0:
                number_detected_annual = len(detected_annual_indices)

                detected[detected_annual_indices] = True

                total_malignant_detected += np.sum(malignant[detected_annual_indices])
                total_benign_detected += number_detected_annual - np.sum(malignant[detected_annual_indices])
                benign_newdetected_count += number_detected_annual - np.sum(malignant[detected_annual_indices])
                benign_total_count += number_detected_annual - np.sum(malignant[detected_annual_indices])
                malignant_newdetected_count += np.sum(malignant[detected_annual_indices])
                malignant_total_count += np.sum(malignant[detected_annual_indices])

                true_positives += np.sum(malignant[detected_annual_indices]) + np.sum(~malignant[detected_annual_indices])

                random_cure_offset = np.random.uniform(CURE_TIME_MIN, CURE_TIME_MAX, size=len(detected_annual_indices))
                cure_year[detected_annual_indices] = year + random_cure_offset

                malignant_at_detection = malignant[detected_annual_indices]

                under_cure_malignant[detected_annual_indices[malignant_at_detection]] = True
                under_cure_benign[detected_annual_indices[~malignant_at_detection]] = True

                # Cost and count these as follow-ups
                followups += number_detected_annual
                cost_followup_total += number_detected_annual * cost_MDT


        # New cancers
        if year < years:
            new_cancers = np.where(np.random.rand(len(no_cancer_indices := np.where(in_program & ~has_cancer)[0])) < annual_incidence)[0]
            newly_cancerous = no_cancer_indices[new_cancers]
            has_cancer[newly_cancerous] = True
            tumor_size[newly_cancerous] = INITIAL_TUMOUR_SIZE
            tumour_start_time[newly_cancerous] = year
            malignant[newly_cancerous] = False
            assign_exponential_growth(newly_cancerous)

        # Recalculate mortality
        if np.sum(in_program) > 0:
            #replace for-loop above with vectorized version
            mortality_status = np.where(in_program & has_cancer & malignant, mort_malignant, mortality_status)
            mortality_status = np.where(in_program & has_cancer & ~malignant, mort_benign, mortality_status)
            mortality_status = np.where(in_program & ~has_cancer, np.vectorize(mort_other_dict.get)(ages), mortality_status)


        # Mortality
        deaths_other = 0
        deaths_benign = 0
        deaths_malignant = 0
        if np.sum(in_program) > 0:
            death_chance = np.random.rand(len(program_indices))
            died = death_chance > mortality_status[program_indices]

            if np.any(died):
                died_indices = program_indices[died]

                deaths_other = np.sum(~has_cancer[died_indices])
                deaths_benign = np.sum(has_cancer[died_indices] & ~malignant[died_indices])
                deaths_malignant = np.sum(has_cancer[died_indices] & malignant[died_indices])

                in_program[died_indices] = False
                # Update everything for those who died
                has_cancer[died_indices] = False
                detected[died_indices] = False
                malignant[died_indices] = False
                tumour_growth_factor[died_indices] = 1.0
                tumour_start_time[died_indices] = np.nan
                cure_year[died_indices] = np.nan
                under_cure_malignant[died_indices] = False
                under_cure_benign[died_indices] = False
                mortality_status[died_indices] = 1.0

        if yearly_print:
            screeningtype_string = '' if screening_on else 'None'
            if screening_on:
                screeningtype_string += 'AI ' if screening_AI else 'Radiologists'
                if screening_AI:
                    screeningtype_string += '2-stage' if ai_stages==2 else '1-stage'
            print(f"Year: {year}, Screening Type: {screeningtype_string}")
            print(f"{np.sum(in_program)} alive, {np.sum(has_cancer)} with cancer, {np.sum(detected)} detected, {np.sum(malignant)} malignant, {np.sum(under_cure_malignant)} under cure malignant, {np.sum(under_cure_benign)} under cure benign, {followups_this_screen} follow-ups")
            print('Cause of death:')
            print('  - Natural causes:', deaths_other)
            print('  - Benign cancer:', deaths_benign)
            print('  - Malignant cancer:', deaths_malignant)
            print('-' * 80)
        deaths_other_by_year.append(deaths_other)
        deaths_benign_by_year.append(deaths_benign)
        deaths_malignant_by_year.append(deaths_malignant)

        life_years_during_program += np.sum(in_program)
        ages += 1



    # After simulation ends
    total_life_years = life_years_during_program
    total_cost = cost_screening_total + cost_followup_total + cost_detection_noscreen_total + cost_treatment_total

    results = {
        'total_cost': total_cost,
        'total_life_years': total_life_years,
        'deaths_other': sum(deaths_other_by_year),
        'deaths_benign': sum(deaths_benign_by_year),
        'deaths_malignant': sum(deaths_malignant_by_year),
        'detected': np.sum(detected),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'followups': followups,
        'malignant_detected': total_malignant_detected,
        'benign_detected': total_benign_detected,
        'cost_screening': cost_screening_total,
        'cost_followup': cost_followup_total,
        'cost_detection_noscreen': cost_detection_noscreen_total,
        'cost_treatment': cost_treatment_total,
        'malig_costs': malignant_costs,
        'benign_costs': benign_costs,
        'benign_detected': benign_total_count,
        'malignant_detected': malignant_total_count
    }
    return results

def evaluate_cost_savings(baseline, intervention):
    cost_diff = intervention['total_cost'] - baseline['total_cost']
    ly_diff = intervention['total_life_years'] - baseline['total_life_years']
    return cost_diff, ly_diff

def pretty_number(value):
    """Format the number with 3 significant figures and scale with K/M/B."""
    abs_val = abs(value)
    suffix = ''
    if abs_val >= 1_000_000_000:
        value = value / 1_000_000_000
        suffix = 'B'
    elif abs_val >= 1_000_000:
        value = value / 1_000_000
        suffix = 'M'
    elif abs_val >= 1_000:
        value = value / 1_000
        suffix = 'K'

    # Use 3 significant figures
    # {:.3g} uses significant figures
    formatted = f"{value:.3g}"
    # In some cases, .3g may produce scientific notation, we accept that as is.
    # It's typically acceptable. If we wanted to strictly avoid scientific notation
    # for certain ranges, we could handle that, but the requirement doesn't forbid it.
    return formatted + suffix

def avg_std_metric(results_list, metric):
    values = [r[metric] for r in results_list]
    return np.mean(values), np.std(values)

def run_multiple_simulations_with_sensitivity(n=100):
    no_screen_results = []
    screen_no_ai_results = []
    screen_ai_results = []
    screen_ai2_results = []

    for i in range(n):
        print(f"Running simulation {i+1}/{n}")
        varied_params = vary_all_params(BASE_PARAMS)
        seed = i
        no_screen = run_simulation(varied_params, screening_on=False, screening_AI=False, seed=seed)
        screen_no_ai = run_simulation(varied_params, screening_on=True, screening_AI=False, seed=seed)
        screen_ai = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1, seed=seed)
        screen_ai2 = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, seed=seed)

        no_screen_results.append(no_screen)
        screen_no_ai_results.append(screen_no_ai)
        screen_ai_results.append(screen_ai)
        screen_ai2_results.append(screen_ai2)

    def scenario_stats(results):
        m_deaths_other, s_deaths_other = avg_std_metric(results, 'deaths_other')
        m_deaths_benign, s_deaths_benign = avg_std_metric(results, 'deaths_benign')
        m_deaths_malignant, s_deaths_malignant = avg_std_metric(results, 'deaths_malignant')

        m_benign_detected, s_benign_detected = avg_std_metric(results, 'benign_detected')
        m_malignant_detected, s_malignant_detected = avg_std_metric(results, 'malignant_detected')

        m_cost_treatment, s_cost_treatment = avg_std_metric(results, 'cost_treatment')

        m_cost_screening, s_cost_screening = avg_std_metric(results, 'cost_screening')
        m_cost_followup, s_cost_followup = avg_std_metric(results, 'cost_followup')
        m_cost_detection_noscreen, s_cost_detection_noscreen = avg_std_metric(results, 'cost_detection_noscreen')

        m_total_cost, s_total_cost = avg_std_metric(results, 'total_cost')
        m_total_ly, s_total_ly = avg_std_metric(results, 'total_life_years')

        m_number_followups, s_number_followups = avg_std_metric(results, 'followups')

        m_malig_costs, s_malig_costs = avg_std_metric(results, 'malig_costs')
        m_benign_costs, s_benign_costs = avg_std_metric(results, 'benign_costs')

        avg_malig_cost = m_malig_costs / m_malignant_detected if m_malignant_detected > 0 else 0
        avg_benign_cost = m_benign_costs / m_benign_detected if m_benign_detected > 0 else 0

        return {
            'deaths_other': (m_deaths_other, s_deaths_other),
            'deaths_benign': (m_deaths_benign, s_deaths_benign),
            'deaths_malignant': (m_deaths_malignant, s_deaths_malignant),
            'benign_detected': (m_benign_detected, s_benign_detected),
            'malignant_detected': (m_malignant_detected, s_malignant_detected),
            'cost_treatment': (m_cost_treatment, s_cost_treatment),
            'cost_screening': (m_cost_screening, s_cost_screening),
            'cost_followup': (m_cost_followup, s_cost_followup),
            'number_followups': (m_number_followups, s_number_followups),
            'cost_detection_noscreen': (m_cost_detection_noscreen, s_cost_detection_noscreen),
            'total_cost': (m_total_cost, s_total_cost),
            'avg_malig_cost': avg_malig_cost,
            'avg_benign_cost': avg_benign_cost,
            'total_life_years': (m_total_ly, s_total_ly)
        }

    no_screen_stats = scenario_stats(no_screen_results)
    screen_no_ai_stats = scenario_stats(screen_no_ai_results)
    screen_ai_stats = scenario_stats(screen_ai_results)
    screen_ai2_stats = scenario_stats(screen_ai2_results)

    diff_cost_no_ai = screen_no_ai_stats['total_cost'][0] - no_screen_stats['total_cost'][0]
    diff_ly_no_ai = screen_no_ai_stats['total_life_years'][0] - no_screen_stats['total_life_years'][0]
    icer_no_ai = diff_cost_no_ai / diff_ly_no_ai if diff_ly_no_ai > 0 else np.inf

    diff_cost_ai = screen_ai_stats['total_cost'][0] - no_screen_stats['total_cost'][0]
    diff_ly_ai = screen_ai_stats['total_life_years'][0] - no_screen_stats['total_life_years'][0]
    icer_ai = diff_cost_ai / diff_ly_ai if diff_ly_ai > 0 else np.inf

    diff_cost_ai2 = screen_ai2_stats['total_cost'][0] - no_screen_stats['total_cost'][0]
    diff_ly_ai2 = screen_ai2_stats['total_life_years'][0] - no_screen_stats['total_life_years'][0]
    icer_ai2 = diff_cost_ai2 / diff_ly_ai2 if diff_ly_ai2 > 0 else np.inf

    def fmt(mean, std):
        # Format with 3 sig figs and scale
        return f"{pretty_number(mean)} ({pretty_number(std)})"

    def print_stats(name, stats):
        print(f"{name} results:")
        print(f"  Total cost: £{fmt(stats['total_cost'][0], stats['total_cost'][1])}")
        print(f"  Total life years: {fmt(stats['total_life_years'][0], stats['total_life_years'][1])}")
        print(f"  Deaths (other): {fmt(stats['deaths_other'][0], stats['deaths_other'][1])}")
        print(f"  Deaths (benign): {fmt(stats['deaths_benign'][0], stats['deaths_benign'][1])}")
        print(f"  Deaths (malignant): {fmt(stats['deaths_malignant'][0], stats['deaths_malignant'][1])}")
        print(f"  Benign detected: {fmt(stats['benign_detected'][0], stats['benign_detected'][1])}")
        print(f"  Malignant detected: {fmt(stats['malignant_detected'][0], stats['malignant_detected'][1])}")
        print(f"  Cost of treatment: £{fmt(stats['cost_treatment'][0], stats['cost_treatment'][1])}")
        print(f"  Cost of screening: £{fmt(stats['cost_screening'][0], stats['cost_screening'][1])}")
        print(f"  Cost of follow-up: £{fmt(stats['cost_followup'][0], stats['cost_followup'][1])}")
        print(f"  Average cost of malignant detection: £{fmt(stats['avg_malig_cost'], 0)}")
        print(f"  Average cost of benign detection: £{fmt(stats['avg_benign_cost'], 0)}")
        print(f"  Number of follow-ups: {fmt(stats['number_followups'][0], stats['number_followups'][1])}")
        print(f"  Cost of detection (no screen): £{fmt(stats['cost_detection_noscreen'][0], stats['cost_detection_noscreen'][1])}")

    print(f"Averaged over {n} simulations with ±20% parameter variation (allowed params only):")
    print_stats("No Screening", no_screen_stats)
    print_stats("Screening (No AI)", screen_no_ai_stats)
    print_stats("Screening (AI) 1 Stage", screen_ai_stats)
    print_stats("Screening (AI) 2 Stage", screen_ai2_stats)

    # For incremental values (not a scenario average), just format them using pretty_number
    incr_cost_no_ai = pretty_number(diff_cost_no_ai)
    incr_ly_no_ai = pretty_number(diff_ly_no_ai)
    incr_icer_no_ai = "∞" if icer_no_ai == np.inf else pretty_number(icer_no_ai)

    incr_cost_ai = pretty_number(diff_cost_ai)
    incr_ly_ai = pretty_number(diff_ly_ai)
    incr_icer_ai = "∞" if icer_ai == np.inf else pretty_number(icer_ai)

    incr_cost_ai2 = pretty_number(diff_cost_ai2)
    incr_ly_ai2 = pretty_number(diff_ly_ai2)
    incr_icer_ai2 = "∞" if icer_ai2 == np.inf else pretty_number(icer_ai2)

    print("Incremental (Screening vs No Screening):")
    print(f"  Screening (No AI) Incremental cost: £{incr_cost_no_ai}, Incremental LY: {incr_ly_no_ai}, ICER: £{incr_icer_no_ai}")
    print(f"  Screening (AI) Incremental cost: £{incr_cost_ai}, Incremental LY: {incr_ly_ai}, ICER: £{incr_icer_ai}")
    print(f"  Screening (AI 2 Stage) Incremental cost: £{incr_cost_ai2}, Incremental LY: {incr_ly_ai2}, ICER: £{incr_icer_ai2}")

# Run the simulations with sensitivity analysis
run_multiple_simulations_with_sensitivity(n=100)

import sys
from utils import *

# -----------------------
# Base Global Parameters
# -----------------------

mort_other_dict = {0: 0.00586, 1: 0.00042, 2: 0.000272, 3: 0.000225, 4: 0.000184, 5: 0.000157, 6: 0.00014,
                   7: 0.000128, 8: 0.000122, 9: 0.000123, 10: 0.000129, 11: 0.000138, 12: 0.000164,
                   13: 0.00022, 14: 0.00031, 15: 0.000446, 16: 0.000637, 17: 0.000868, 18: 0.0011,
                   19: 0.00127, 20: 0.001373, 21: 0.001488, 22: 0.001605, 23: 0.001714, 24: 0.001835,
                   25: 0.001963, 26: 0.002082, 27: 0.002202, 28: 0.00233, 29: 0.002457, 30: 0.002574,
                   31: 0.002683, 32: 0.00279587, 33: 0.002881, 34: 0.002974, 35: 0.003074, 36: 0.003175,
                   37: 0.003295, 38: 0.003444, 39: 0.003608, 40: 0.00378, 41: 0.003958, 42: 0.004144,
                   43: 0.004337, 44: 0.00454, 45: 0.004774, 46: 0.005064, 47: 0.005399, 48: 0.005796,
                   49: 0.006214, 50: 0.006671, 51: 0.007167, 52: 0.007736, 53: 0.008351, 54: 0.009035,
                   55: 0.00977, 56: 0.010567, 57: 0.011398, 58: 0.012291, 59: 0.013224, 60: 0.014267,
                   61: 0.015353, 62: 0.016484, 63: 0.017617, 64: 0.018759, 65: 0.019914, 66: 0.021104,
                   67: 0.022423, 68: 0.023847, 69: 0.025357, 70: 0.02705, 71: 0.02897, 72: 0.031188,
                   73: 0.033754, 74: 0.036747, 75: 0.040563, 76: 0.044308, 77: 0.048498, 78: 0.053229,
                   79: 0.058778, 80: 0.064617, 81: 0.070947, 82: 0.077834, 83: 0.085686, 84: 0.094809,
                   85: 0.10509, 86: 0.116592, 87: 0.129306, 88: 0.142732, 89: 0.157638, 90: 0.174458,
                   91: 0.193027, 92: 0.21293, 93: 0.232657, 94: 0.251826, 95: 0.270943, 96: 0.289756,
                   97: 0.307998, 98: 0.325393, 99: 0.341662, 100: 0.358746, 101: 0.376683, 102: 0.395517,
                   103: 0.415293, 104: 0.436058, 105: 0.45786, 106: 0.480753, 107: 0.504791, 108: 0.530031,
                   109: 0.556532, 110: 0.584359, 111: 0.613577, 112: 0.644256, 113: 0.676468,
                   114: 0.710292, 115: 0.745806, 116: 0.783097, 117: 0.822251, 118: 0.863364,
                   119: 0.906532, 120: 1.0, 121: 1.0, 122: 1.0, 123: 1.0, 124: 1.0,
                   125: 1.0}  # age-standardised rate of death: https://www.ssa.gov/oact/STATS/table4c6.html#fn1

# high-level sanity checks:
#   12-14k new cases/year, 4-5k deaths/year: https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/kidney-cancer

BASE_PARAMS = {

    # Population and simulation (no variation)
    "N": 100_000,  # 20 million people between 45-74
    "years": 20,
    "screen_interval": 2,
    "yearly_print": False,

    # Disease model parameters (no variation)
    "annual_incidence": 14000/27000000,
    # roughly 14k new cases per year https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/kidney-cancer
    # roughly 27 million people aged 45-80 in the UK #https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/articles/ukpopulationpyramidinteractive/2020-01-08
    "initial_prevalence": 0.003,  # 0.3% tumour prevalence on first screen - taken from YKST (0.25%) and increased 20% to account for missed cases by radiologists
    "pct_large": 0.57,  # fraction of initial cancers that are large

    #starting distribution (allowed to vary)
    "mean_size": 22449,  # radius=1.75cm => volume in mm³
    "std_size": 8000,

    # Tumor size thresholds and size-specific progression probabilities
    # (no variation)
    "very_small_threshold": 4188.79,  # radius=1cm => volume
    "large_threshold": 14137,  # radius=1.5cm => volume
    "threshold_1cm": 524,  # approximate volume threshold
    "threshold_2cm": 4188.79,  # approximate volume threshold
    "threshold_3cm": 14137,  # approximate volume threshold
    "threshold_4cm": 33510,  # radius=2cm => volume
    "threshold_5cm": 65450,
    "threshold_6cm": 113097,
    "prob_highgrade_at_1cm": 0.16,
    "prob_highgrade_at_2cm": 0.17,
    "prob_highgrade_at_3cm": 0.27,
    "prob_highgrade_at_4cm": 0.33,
    "prob_highgrade_at_5cm": 0.4,
    "prob_highgrade_at_over5cm": 0.6,
    # probability of high grade % each cm: https://www.auajournals.org/doi/epdf/10.1016/j.juro.2009.01.027

    # Growth parameters (no variation)
    "doubling_time_mean": 1.5,  # https://pmc.ncbi.nlm.nih.gov/articles/PMC3989814/
    "doubling_time_sd": 0.5,

    # Costs (no variation)
    "cost_screen_ai": 15,  # assume overhead, contrast, etc. is absorbed. Use same as Sheffield for unit incremental cost of abd acquistion
    "cost_screen_no_ai": 30,  # cost_screen_ai + £15 for radiologist
    "cost_followup_ai": 35,  # cost_screen_ai + £20 for contrast
    "cost_followup_no_ai": 150,  # no AI follow-up cost, therefore assumed in hospital outpatients
    "cost_MDT": 230,  # multidisciplinary team cost
    "cost_second_review_cect": 50,  # cost of second review of CECT-AI positive Rosenberg, Christian, et al. "Teleradiology from the provider’s perspective—Cost analysis for a mid-size university hospital." European radiology 23 (2013): 2197-2205.

    # Annual detection probabilities (allowed to vary)
    "annual_detection_prob_lowgrade": 0.12,  # scaled so no-screen produces correct ratio of cases in the no-screen long-term
    "annual_detection_prob_highgrade": 0.5,  # scaled likewise
    # relative number of high-grade to low-grade detections is roughly 8:11 in UK - https://nhsd-ndrs.shinyapps.io/incidence_and_mortality/


    # Mortality every year, copy the values from above and 1-value for each year
    # (no variation)
    "mort_other_dict": {k: 1 - v for k, v in mort_other_dict.items()},
    "starting_age_min": 55,
    "starting_age_max": 75,
    "stop_screening_age": 75,
    "start_screening_age": 55,

    # Lifeyears saved by no progression (allowed to vary)
    "lifeyears_saved_by_noprogress":9.7,

    # Initial tumour size for new cancers (allowed to vary)
    "INITIAL_TUMOUR_SIZE": 526,

    # Screening detection parameters (no variation)
    "det_params_no_screen": {
        'det_prob_very_small': 0.0,
        'det_prob_small': 0.0,
        'det_prob_large': 0.0,
        'specificity': 1.0,
        'onsite_followup': False
    },
    "det_params_screen_no_ai": {
        'det_prob_very_small': 0.171, #an estimate - assumed detection sensivity linearly scales with size (0.5355- (0.9-0.5355)/(3-2)) = 0.171
        'det_prob_small': 0.5355, #average of consultant uroradiologist across LDCT and NCCT IT
        'det_prob_large': 0.90, #average of consultant uroradiologist across LDCT and NCCT IT
        'specificity': 0.955, #observed in YKST
        'onsite_followup': False
    },
    "det_params_screen_ai_2stage": {
        'det_prob_very_small': 0.171/3, # In YKST, we detected 2/6 known masses less than 20mm, so one third of radiologist-detected
        'det_prob_small': 0.6785, #average of AI across LDCT and NCCT IT
        'det_prob_large': 0.978, #average of AI across LDCT and NCCT IT
        'specificity': 0.808, #observed in YKST
        'second_specificity': 0.7779, # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True
    },
    "det_params_screen_ai_2stage_extrasens": { # AI with extra sensitivity for very small - same as radiologist
        'det_prob_very_small': 0.171,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.7779,
        'onsite_followup': True
    },
    "det_params_screen_ai_1stage": {
        'det_prob_very_small': 0.171/3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'onsite_followup': False
    }
}

# Parameters not to vary
NO_VARIATION_KEYS = {
    "N", "years", "screen_interval","annual_incidence", "initial_prevalence", "pct_large", "doubling_time_mean", "doubling_time_sd",
    "mort_other_dict", "starting_age","starting_age_min", "starting_age_max", "stop_screening_age", "start_screening_age",
    "det_params_no_screen", "det_params_screen_no_ai", "det_params_screen_ai_2stage",
    "det_params_screen_ai_2stage_extrasens", "det_params_screen_ai_1stage",
    "cost_screen_ai", "cost_screen_no_ai", "cost_followup_ai", "cost_followup_no_ai", "cost_MDT", "cost_second_review_cect",
    "very_small_threshold", "large_threshold", "threshold_1cm", "threshold_2cm", "threshold_3cm", "threshold_4cm",
    "threshold_5cm", "threshold_6cm", "prob_highgrade_at_1cm", "prob_highgrade_at_2cm", "prob_highgrade_at_3cm",
    "prob_highgrade_at_4cm", "prob_highgrade_at_5cm", "prob_highgrade_at_over5cm",
}


def run_simulation(params, screening_on, screening_AI, ai_stages=2,
                   extrasens=False,seed=None):
    if seed is not None:
        np.random.seed(seed)
    if not screening_on:
        det = params["det_params_no_screen"]
    elif screening_AI and ai_stages == 2:
        if extrasens:
            det = params["det_params_screen_ai_2stage_extrasens"]
        else:
            det = params["det_params_screen_ai_2stage"]
        second_specificity = det['second_specificity']
    elif screening_AI and ai_stages == 1:
        det = params["det_params_screen_ai_1stage"]
    else:
        det = params["det_params_screen_no_ai"]

    # initialise variables
    det_prob_very_small = det['det_prob_very_small']
    det_prob_small = det['det_prob_small']
    det_prob_large = det['det_prob_large']
    specificity = det['specificity']
    onsite_followup = det['onsite_followup']
    N = params["N"]
    years = params["years"]
    screen_interval = params["screen_interval"]
    starting_age_min = params["starting_age_min"]
    starting_age_max = params["starting_age_max"]
    stop_screening_age = params["stop_screening_age"]  # when age>75, remove from screening
    start_screening_age = params["start_screening_age"]  # when age>55, add to screening
    lifeyears_saved_by_noprogress = params["lifeyears_saved_by_noprogress"]
    mort_other_dict = params["mort_other_dict"]
    ages = np.random.randint(starting_age_min, starting_age_max + 1,
                             N)  # ages is an a uniformly distributed array of ages between starting_age_min and starting_age_max
    annual_incidence = params["annual_incidence"]
    initial_prevalence = params["initial_prevalence"]
    pct_large = params["pct_large"]
    mean_size = params["mean_size"]
    std_size = params["std_size"]
    very_small_threshold = params["very_small_threshold"]
    large_threshold = params["large_threshold"]
    threshold_1cm = params["threshold_1cm"]
    threshold_2cm = params["threshold_2cm"]
    threshold_3cm = params["threshold_3cm"]
    threshold_4cm = params["threshold_4cm"]
    threshold_5cm = params["threshold_5cm"]
    threshold_6cm = params["threshold_6cm"]
    prob_highgrade_at_1cm = params["prob_highgrade_at_1cm"]
    prob_highgrade_at_2cm = params["prob_highgrade_at_2cm"]
    prob_highgrade_at_3cm = params["prob_highgrade_at_3cm"]
    prob_highgrade_at_4cm = params["prob_highgrade_at_4cm"]
    prob_highgrade_at_5cm = params["prob_highgrade_at_5cm"]
    prob_highgrade_at_over5cm = params["prob_highgrade_at_over5cm"]
    doubling_time_mean = params["doubling_time_mean"]
    doubling_time_sd = params["doubling_time_sd"]
    cost_screen_ai = params["cost_screen_ai"]
    cost_screen_no_ai = params["cost_screen_no_ai"]
    cost_followup_ai = params["cost_followup_ai"]
    cost_followup_no_ai = params["cost_followup_no_ai"]
    cost_second_review_cect = params["cost_second_review_cect"]
    cost_MDT = params["cost_MDT"]
    annual_detection_prob_lowgrade = params["annual_detection_prob_lowgrade"]
    annual_detection_prob_highgrade = params["annual_detection_prob_highgrade"]
    INITIAL_TUMOUR_SIZE = params["INITIAL_TUMOUR_SIZE"]

    # initialise tracking arrays
    screen_years = set(range(0, years + 1, screen_interval))
    in_program = np.ones(N, dtype=bool)
    age_at_diagnosis = np.full(N, np.nan)
    has_cancer = np.zeros(N, dtype=bool)
    tumor_size = np.zeros(N, dtype=float)
    detected = np.zeros(N, dtype=bool)
    highgrade = np.zeros(N, dtype=bool)
    tumour_start_time = np.full(N, np.nan)
    tumour_growth_factor = np.ones(N)
    mortality_status = np.vectorize(mort_other_dict.get)(ages)
    tumour_detected_time = np.full(N, np.nan)  # the simulation year in which cancer is first detected
    highgrade_at_detection = np.zeros(N, dtype=bool)
    year_of_death = np.full(N, np.nan)
    deaths_other_by_year = []
    deaths_lowgrade_by_year = []
    deaths_highgrade_by_year = []

    # initialise all counts
    false_positives = 0
    true_positives = 0
    followups = 0
    cost_screening_total = 0.0
    cost_followup_total = 0.0
    cost_detection_noscreen_total = 0.0
    highgrade_costs = 0.0
    lowgrade_costs = 0.0
    total_highgrade_detected = 0
    total_lowgrade_detected = 0

    # initialise starting cancers
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
        tumour_growth_factor = assign_exponential_growth(initial_cancer_indices, doubling_time_mean, doubling_time_sd,
                                                         tumour_growth_factor)
        thresholds = [threshold_1cm, threshold_2cm, threshold_3cm, threshold_4cm, threshold_5cm, threshold_6cm]
        probabilties = [prob_highgrade_at_1cm, prob_highgrade_at_2cm, prob_highgrade_at_3cm, prob_highgrade_at_4cm,
                        prob_highgrade_at_5cm, prob_highgrade_at_over5cm]
        highgrade = update_hgnancy_on_size_crossing_multithresh(np.zeros(num_initial_cancers),
                                                                initial_sizes,
                                                                initial_cancer_indices,
                                                                highgrade,
                                                                thresholds,
                                                                probabilties)

    # main simulation loop
    for year in range(0, years + 1):
        program_indices = np.where(in_program)[0]
        highgrade_newdetected_count = 0
        lowgrade_newdetected_count = 0

        # Growth
        cancer_indices = np.where(in_program & has_cancer)[0]  # Corrected
        if len(cancer_indices) > 0:
            prev_sizes = tumor_size[cancer_indices].copy()
            tumor_size[cancer_indices] *= tumour_growth_factor[cancer_indices]
            new_sizes = tumor_size[cancer_indices]
            thresholds = [threshold_1cm, threshold_2cm, threshold_3cm, threshold_4cm, threshold_5cm]
            probabilities = [prob_highgrade_at_1cm, prob_highgrade_at_2cm, prob_highgrade_at_3cm, prob_highgrade_at_4cm,
                             prob_highgrade_at_over5cm]
            highgrade = update_hgnancy_on_size_crossing_multithresh(prev_sizes,
                                                                    new_sizes,
                                                                    cancer_indices,
                                                                    highgrade,
                                                                    thresholds,
                                                                    probabilities)

        # Screening detection
        followups_this_screen = 0
        if screening_on and year in screen_years:
            program_indices = np.where(in_program)[0]  # Define program_indices correctly
            in_screening = in_program & (ages <= stop_screening_age) & (ages >= start_screening_age)

            if screening_AI:
                cost_screening_total += np.sum(in_screening) * cost_screen_ai
            else:
                cost_screening_total += np.sum(in_screening) * cost_screen_no_ai
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
                #remove any detected cancers that are not meant to be in_screening
                detection_array[~in_screening[cancer_indices]] = False
                detected_cancer_indices = cancer_indices[detection_array]

                tumour_detected_time[detected_cancer_indices] = year
                highgrade_at_detection[detected_cancer_indices] = highgrade[detected_cancer_indices]
                number_detected_this_screen = np.sum(detection_array)
                detected_cancer_indices = cancer_indices[detection_array]
                highgrade_detected_count = np.sum(highgrade[detected_cancer_indices])
                lowgrade_newdetected_count += np.sum(~highgrade[detected_cancer_indices])
                total_lowgrade_detected += np.sum(~highgrade[detected_cancer_indices])
                highgrade_newdetected_count += highgrade_detected_count
                total_highgrade_detected += highgrade_detected_count
                true_positives += number_detected_this_screen
                in_screening_nocancer = in_screening[~has_cancer]
                false_detections = np.random.rand(len(in_screening_nocancer)) < (1 - specificity)
                false_positives += np.sum(false_detections)
                followups_this_screen = number_detected_this_screen + np.sum(false_detections)

                detected[detected_cancer_indices] = True
                tumor_size[detected_cancer_indices] = 0.0
                # remove from program
                in_program[detected_cancer_indices] = False
            else:
                followups_this_screen = 0
                number_detected_this_screen = 0

            # Follow-up cost
            if followups_this_screen > 0:
                followups += followups_this_screen
                if onsite_followup:
                    cost_followup_total += followups_this_screen * cost_followup_ai
                    # use second specificity to assess who must be sent for second review
                    # all positives after CECT AI must be checked by a radiologist
                    n_second_reviews = np.sum(np.random.rand(followups_this_screen) > second_specificity)
                    cost_followup_total += n_second_reviews * cost_second_review_cect
                else:
                    cost_followup_total += followups_this_screen * cost_followup_no_ai

            # Add MDT cost to all screen-detected tumors
            cost_followup_total += number_detected_this_screen * cost_MDT
            lowgrade_costs += lowgrade_newdetected_count * cost_MDT
            highgrade_costs += highgrade_newdetected_count * cost_MDT

        # Annual detection for all non-screened cancers - serendipitous detection
        undetected_cancer_indices = np.where(in_program & ~detected & has_cancer)[0]
        if len(undetected_cancer_indices) > 0:
            highgrade_undetected_indices = undetected_cancer_indices[highgrade[undetected_cancer_indices]]
            lowgrade_undetected_indices = undetected_cancer_indices[~highgrade[undetected_cancer_indices]]

            detected_lowgrade_indices = lowgrade_undetected_indices[
                np.random.rand(len(lowgrade_undetected_indices)) < annual_detection_prob_lowgrade] if len(
                lowgrade_undetected_indices) > 0 else np.array([], dtype=int)

            detected_highgrade_indices = highgrade_undetected_indices[
                np.random.rand(len(highgrade_undetected_indices)) < annual_detection_prob_highgrade] if len(
                highgrade_undetected_indices) > 0 else np.array([], dtype=int)

            detected_annual_indices = np.concatenate((detected_lowgrade_indices, detected_highgrade_indices))
            if len(detected_annual_indices) > 0:
                number_detected_annual = len(detected_annual_indices)
                detected[detected_annual_indices] = True
                age_at_diagnosis[detected_annual_indices] = ages[detected_annual_indices]
                total_highgrade_detected += np.sum(highgrade[detected_annual_indices])
                total_lowgrade_detected += number_detected_annual - np.sum(highgrade[detected_annual_indices])
                lowgrade_newdetected_count += number_detected_annual - np.sum(highgrade[detected_annual_indices])
                highgrade_newdetected_count += np.sum(highgrade[detected_annual_indices])
                highgrade_at_detection[detected_annual_indices] = highgrade[detected_annual_indices]
                tumour_detected_time[detected_annual_indices] = year
                true_positives += np.sum(highgrade[detected_annual_indices]) + np.sum(
                    ~highgrade[detected_annual_indices])
                # remove from program
                in_program[detected_annual_indices] = False

                followups += number_detected_annual
                cost_followup_total += number_detected_annual * cost_MDT


        # New cancer formation
        if year < years:
            new_cancers = np.where(
                np.random.rand(len(no_cancer_indices := np.where(in_program & ~has_cancer)[0])) < annual_incidence)[0]
            newly_cancerous = no_cancer_indices[new_cancers]
            if len(newly_cancerous) > 0:
                has_cancer[newly_cancerous] = True
                tumor_size[newly_cancerous] = INITIAL_TUMOUR_SIZE
                tumour_start_time[newly_cancerous] = year
                highgrade[newly_cancerous] = False
                assign_exponential_growth(newly_cancerous, doubling_time_mean, doubling_time_sd, tumour_growth_factor)

        # Reassign changed mortality statuses based purely on age and ccount for non-cancer deaths
        if np.sum(in_program) > 0:
            mortality_status = np.where(in_program, np.vectorize(mort_other_dict.get)(ages),
                                        mortality_status)
            death_chance = np.random.rand(len(program_indices))
            died = death_chance > mortality_status[program_indices]
            if np.any(died):
                died_indices = program_indices[died]
                in_program[died_indices] = False
                year_of_death[died_indices] = year

        ages += 1

    total_cost = (cost_screening_total
                  + cost_followup_total
                  + cost_detection_noscreen_total)



    results = {
        'total_cost': total_cost,
        'deaths_other': sum(deaths_other_by_year),
        'deaths_lowgrade': sum(deaths_lowgrade_by_year),
        'deaths_highgrade': sum(deaths_highgrade_by_year),
        'detected': np.sum(detected),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'followups': followups,
        'highgrade_detected': total_highgrade_detected,
        'lowgrade_detected': total_lowgrade_detected,
        'cost_screening': cost_screening_total,
        'cost_followup': cost_followup_total,
        'cost_detection_noscreen': cost_detection_noscreen_total,
        'hg_costs': highgrade_costs,
        'lowgrade_costs': lowgrade_costs,
    }
    return results


def run_multiple_simulations_with_sensitivity(n=100):
    no_screen_results = []
    screen_no_ai_results = []
    screen_ai_results = []
    screen_ai2_results = []
    screen_ai2_extra_results = []

    for i in range(n):
        print(f"Running simulation {i + 1}/{n}")
        varied_params = vary_all_params(BASE_PARAMS, NO_VARIATION_KEYS)
        seed = i
        no_screen = run_simulation(varied_params, screening_on=False, screening_AI=False, seed=seed)
        screen_no_ai = run_simulation(varied_params, screening_on=True, screening_AI=False, seed=seed)
        screen_ai = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1, seed=seed)
        screen_ai2 = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, seed=seed)
        screen_ai2_extra = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, extrasens=True, seed=seed)

        no_screen_results.append(no_screen)
        screen_no_ai_results.append(screen_no_ai)
        screen_ai_results.append(screen_ai)
        screen_ai2_results.append(screen_ai2)
        screen_ai2_extra_results.append(screen_ai2_extra)

    no_screen_stats = scenario_stats_notreat(no_screen_results)
    screen_no_ai_stats = scenario_stats_notreat(screen_no_ai_results)
    screen_ai_stats = scenario_stats_notreat(screen_ai_results)
    screen_ai2_stats = scenario_stats_notreat(screen_ai2_results)
    screen_ai2_extra_stats = scenario_stats_notreat(screen_ai2_extra_results)
    lifeyears_saved_by_noprogression = BASE_PARAMS["lifeyears_saved_by_noprogress"]

    #find the differences of highgrade_detected in each screening arm with no screening, call it A1, A2, A3
    #find the differences of lowgrade_detected in each screening arm with no screening, call it B1, B2, B3

    A1 = no_screen_stats['highgrade_detected'][0] -  screen_no_ai_stats['highgrade_detected'][0]
    A2 = no_screen_stats['highgrade_detected'][0] - screen_ai_stats['highgrade_detected'][0]
    A3 = no_screen_stats['highgrade_detected'][0] - screen_ai2_stats['highgrade_detected'][0]
    A4 = no_screen_stats['highgrade_detected'][0] - screen_ai2_extra_stats['highgrade_detected'][0]

    B1 = screen_no_ai_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B2 = screen_ai_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B3 = screen_ai2_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B4 = screen_ai2_extra_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]

    # find average of each and report as 'Progressions Prevented'
    no_screen_stats['Progressions Prevented'] = 0
    screen_no_ai_stats['Progressions Prevented'] = (A1 + B1)/2
    screen_ai_stats['Progressions Prevented'] = (A2 + B2)/2
    screen_ai2_stats['Progressions Prevented'] = (A3 + B3)/2
    screen_ai2_extra_stats['Progressions Prevented'] = (A4 + B4)/2

    no_screen_stats['Cost-Effectiveness'] = 0
    screen_no_ai_stats['Cost-Effectiveness'] = (screen_no_ai_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_no_ai_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_stats['Cost-Effectiveness'] = (screen_ai_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai2_stats['Cost-Effectiveness'] = (screen_ai2_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai2_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai2_extra_stats['Cost-Effectiveness'] = (screen_ai2_extra_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai2_extra_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)

    print(f"Averaged over {n} simulations with ±20% parameter variation (allowed params only):")
    print_stats_notreat("No Screening", no_screen_stats)
    print_stats_notreat("Screening (No AI)", screen_no_ai_stats)
    print_stats_notreat("Screening (AI) 1 Stage", screen_ai_stats)
    print_stats_notreat("Screening (AI) 2 Stage", screen_ai2_stats)
    print_stats_notreat("Screening (AI) 2 Stage Extra Sensitive", screen_ai2_extra_stats)

run_multiple_simulations_with_sensitivity(n=1000)
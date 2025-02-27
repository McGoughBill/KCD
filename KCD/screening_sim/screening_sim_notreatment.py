import sys
from utils import *
import csv
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
    "cost_screen_ai": 20,  # assume overhead, contrast, etc. is absorbed. Use same as Sheffield for unit incremental cost of abd acquistion
    # + £5 for AI
    "cost_screen_no_ai": 30,  # cost_screen_ai + £15 for radiologist
    "cost_followup_ai": 40,  # cost_screen_ai + £20 for contrast
    "cost_followup_no_ai": 150,  # no AI follow-up cost, therefore assumed in hospital outpatients
    "cost_MDT": 230,  # multidisciplinary team cost
    "cost_second_review_cect": 50,  # cost of second review of CECT-AI positive Rosenberg, Christian, et al. "Teleradiology from the provider’s perspective—Cost analysis for a mid-size university hospital." European radiology 23 (2013): 2197-2205.
    "cost_followup_rad_ncct": 15,  # ncct_reporting_cost = cost_screen_no_ai - (cost_screen_ai - 5 from no AI)


    # Annual detection probabilities (allowed to vary)
    "annual_detection_prob_lowgrade": 0.0724,  # use sojourn time of 5.7 years
    "annual_detection_prob_highgrade": 0.2353,  # 11/19 cancers detected are high grade, but only 11/37 resident cancers are high grade
    # relative number of high-grade to low-grade detections is roughly 8:11 in UK - https://nhsd-ndrs.shinyapps.io/incidence_and_mortality/

    "dose_screen":4, #mGy
    "dose_followup":10, #mGy

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
        'onsite_followup': False,
        'rad_followup': False,
        'skip_cect': False,
        'triage_skip': False
    },
    "det_params_screen_no_ai": {
        'det_prob_very_small': 0.171, #an estimate - assumed detection sensivity linearly scales with size (0.5355- (0.9-0.5355)/(3-2)) = 0.171
        'det_prob_small': 0.5355, #average of consultant uroradiologist across LDCT and NCCT IT
        'det_prob_large': 0.90, #average of consultant uroradiologist across LDCT and NCCT IT
        'specificity': 0.955, #observed in YKST
        'onsite_followup': False,
        'rad_followup': False,
        'skip_cect': False,
        'triage_skip': False
    },
    "det_params_screen_ai_2stage": {
        'det_prob_very_small': 0.171/3, # In YKST, we detected 2/6 known masses less than 20mm, so one third of radiologist-detected
        'det_prob_small': 0.6785, #average of AI across LDCT and NCCT IT
        'det_prob_large': 0.978, #average of AI across LDCT and NCCT IT
        'specificity': 0.808, #observed in YKST
        'second_specificity': 0.7779, # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': False,
        'triage_skip': False
    },
    "det_params_screen_ai_2stage_extrasens": { # AI with extra sensitivity for very small - same as radiologist
        'det_prob_very_small': 0.171,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.7779,
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': False,
        'triage_skip': False
    },
    "det_params_screen_ai_1stage_radFU": {  # AI with rad follow-up on NCCT
        'det_prob_very_small': 0.171/3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.8824**2, # is taken from internal testing
        'onsite_followup': False,
        'rad_followup': True,
        'skip_cect': False,
        'triage_skip': False
    },
    "det_params_screen_ai_2stage_withskip": {
        'det_prob_very_small': 0.171 / 3,
        # In YKST, we detected 2/6 known masses less than 20mm, so one third of radiologist-detected
        'det_prob_small': 0.6785,  # average of AI across LDCT and NCCT IT
        'det_prob_large': 0.978,  # average of AI across LDCT and NCCT IT
        'specificity': 0.808,  # observed in YKST
        'second_specificity': 0.7779,  # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': True,
        'triage_skip': False
    },
    "det_params_screen_ai_2stage_withskip_triaging": {
        'triage_skip_very_small': 0.25,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.447,  # observed in YKST

        'det_prob_very_small': 0.171 / 3,
        # In YKST, we detected 2/6 known masses less than 20mm, so one third of radiologist-detected
        'det_prob_small': 0.6785,  # average of AI across LDCT and NCCT IT
        'det_prob_large': 0.978,  # average of AI across LDCT and NCCT IT
        'specificity': 0.808,  # observed in YKST
        'second_specificity': 0.7779,  # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': True,
        'triage_skip': True
    },
    "det_params_screen_ai_2stage_extrasens_withskip": {  # AI with extra sensitivity for very small - same as radiologist
        'det_prob_very_small': 0.171,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.7779,
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': True,
        'triage_skip': False
    },
    "det_params_screen_ai_1stage": {
        'det_prob_very_small': 0.171/3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'onsite_followup': False,
        'rad_followup': False,
        'skip_cect': False,
        'triage_skip': False
    },
    "det_params_screen_ai_1stage_skiptriage": {
        'triage_skip_very_small': 0.25, #refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0, #refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0, #refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.447, #observed in YKST
        'det_prob_very_small': 0.171 / 3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'onsite_followup': False,
        'rad_followup': False,
        'skip_cect': False,
        'triage_skip': True
    },
    "det_params_screen_ai_1stage_skiptriage_radfu": {
        'triage_skip_very_small': 0.25,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.447,  # observed in YKST

        'det_prob_very_small': 0.171 / 3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.8824 ** 2,  # is taken from internal testing
        'onsite_followup': False,
        'rad_followup': True,
        'skip_cect': False,
        'triage_skip': True
    },
    "det_params_screen_ai_1stage_skiptriage_radfu_extrasensitive": {
        'triage_skip_very_small': 0,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.447,  # observed in YKST

        'det_prob_very_small': 0.171 ,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.8824 ** 2,  # is taken from internal testing
        'onsite_followup': False,
        'rad_followup': True,
        'skip_cect': False,
        'triage_skip': True
    },
    "det_params_screen_ai_2stage_skiptriage_extrasensitive": {
        'triage_skip_very_small': 0,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.447,  # observed in YKST

        'det_prob_very_small': 0.171,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.808,
        'second_specificity': 0.7779,  # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': True,
        'triage_skip': True
    },
    "det_params_screen_ai_2stage_withskip_triage_extraspecific": {
        'triage_skip_very_small': 0.25,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.55,  # observed in YKST

        'det_prob_very_small': 0.171/3,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.9,
        'second_specificity': 0.9,  # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': True,
        'triage_skip': True
    },
    "det_params_screen_ai_1stage_skiptriage_radfu_extraspecific": {
        'triage_skip_very_small': 0.25,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.55,  # observed in YKST

        'det_prob_very_small': 0.171/3 ,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.9,
        'second_specificity': 0.8824 ** 2,  # is taken from internal testing
        'onsite_followup': False,
        'rad_followup': True,
        'skip_cect': False,
        'triage_skip': True
    },
    "det_params_screen_ai_2stage_withskip_triage_extraspecific_extrasens": {
        'triage_skip_very_small': 0.0,
        # refers to the proportion of the missed that are told to skip next screen - taken from YKST
        'triage_skip_small': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_skip_large': 0,  # refers to the proportion of these missed that are told to skip next screen
        'triage_specificity': 0.55,  # observed in YKST

        'det_prob_very_small': 0.171,
        'det_prob_small': 0.6785,
        'det_prob_large': 0.978,
        'specificity': 0.9,
        'second_specificity': 0.7779,  # per patient specificity of CECT-AI, observed 0.882 per kidney in IT
        'onsite_followup': True,
        'rad_followup': False,
        'skip_cect': True,
        'triage_skip': True
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
    "prob_highgrade_at_4cm", "prob_highgrade_at_5cm", "prob_highgrade_at_over5cm","cost_followup_rad_ncct"
}


def run_simulation(params, screening_on, screening_AI, ai_stages=2,
                   with_skip=False,extrasens=False,seed=None,
                   rad_followup=False,triage_skip=False,extraspec=False):
    if seed is not None:
        np.random.seed(seed)
    if not screening_on:
        print('A')
        det = params["det_params_no_screen"]
    elif screening_AI and ai_stages == 2:
        if extrasens:
            if with_skip and not triage_skip:
                print('G')
                det = params["det_params_screen_ai_2stage_extrasens_withskip"]
            elif triage_skip and not with_skip:
                print('L')
                det = params["det_params_screen_ai_2stage_skiptriage_extrasensitive"]
                triage_skip_very_small = det['triage_skip_very_small']
                triage_skip_small = det['triage_skip_small']
                triage_skip_large = det['triage_skip_large']
                triage_specificity = det['triage_specificity']
            elif extraspec and with_skip and triage_skip:
                print('P')
                det = params["det_params_screen_ai_2stage_withskip_triage_extraspecific_extrasens"]
                triage_skip_very_small = det['triage_skip_very_small']
                triage_skip_small = det['triage_skip_small']
                triage_skip_large = det['triage_skip_large']
                triage_specificity = det['triage_specificity']
            else:
                print('E')
                det = params["det_params_screen_ai_2stage_extrasens"]
        elif extraspec:
            # for det_params_screen_ai_2stage_skiptriage_extraspecific
            if triage_skip and with_skip:
                print('N')
                det = params["det_params_screen_ai_2stage_withskip_triage_extraspecific"]
                triage_skip_very_small = det['triage_skip_very_small']
                triage_skip_small = det['triage_skip_small']
                triage_skip_large = det['triage_skip_large']
                triage_specificity = det['triage_specificity']
            else:
                raise ValueError("extraspec only works with triage_skip and skip_cect if ai_stages=2")
        else:
            if with_skip and not triage_skip:
                print('F')
                det = params["det_params_screen_ai_2stage_withskip"]
            elif triage_skip and with_skip:
                print('M')
                det = params["det_params_screen_ai_2stage_withskip_triaging"]
                triage_skip_very_small = det['triage_skip_very_small']
                triage_skip_small = det['triage_skip_small']
                triage_skip_large = det['triage_skip_large']
                triage_specificity = det['triage_specificity']
            else:
                print('D')
                det = params["det_params_screen_ai_2stage"]
        second_specificity = det['second_specificity']
    elif screening_AI and ai_stages == 1:
        if rad_followup and not triage_skip:
            print('H')
            det = params["det_params_screen_ai_1stage_radFU"]
            second_specificity = det['second_specificity']
        elif rad_followup and triage_skip and extraspec:
            print('O')
            det = params["det_params_screen_ai_1stage_skiptriage_radfu_extraspecific"]
            triage_skip_very_small = det['triage_skip_very_small']
            triage_skip_small = det['triage_skip_small']
            triage_skip_large = det['triage_skip_large']
            triage_specificity = det['triage_specificity']
            second_specificity = det['second_specificity']
        elif not rad_followup and triage_skip:
            print('I')
            det = params["det_params_screen_ai_1stage_skiptriage"]
            triage_skip_very_small = det['triage_skip_very_small']
            triage_skip_small = det['triage_skip_small']
            triage_skip_large = det['triage_skip_large']
            triage_specificity = det['triage_specificity']
        elif rad_followup and triage_skip and not extrasens:
            print('J')
            det = params["det_params_screen_ai_1stage_skiptriage_radfu"]
            triage_skip_very_small = det['triage_skip_very_small']
            triage_skip_small = det['triage_skip_small']
            triage_skip_large = det['triage_skip_large']
            triage_specificity = det['triage_specificity']
            second_specificity = det['second_specificity']
        elif rad_followup and triage_skip and extrasens:
            print('K')
            det = params["det_params_screen_ai_1stage_skiptriage_radfu_extrasensitive"]
            triage_skip_very_small = det['triage_skip_very_small']
            triage_skip_small = det['triage_skip_small']
            triage_skip_large = det['triage_skip_large']
            triage_specificity = det['triage_specificity']
            second_specificity = det['second_specificity']
        else:
            print('C')
            det = params["det_params_screen_ai_1stage"]
    else:
        print('B')
        det = params["det_params_screen_no_ai"]

    # initialise variables
    det_prob_very_small = det['det_prob_very_small']
    det_prob_small = det['det_prob_small']
    det_prob_large = det['det_prob_large']
    det_skip_cect = det['skip_cect']
    rad_followup = det['rad_followup']
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
    cost_followup_rad_ncct = params["cost_followup_rad_ncct"]
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
    skip_next = np.zeros(N, dtype=bool)
    deaths_other_by_year = []
    deaths_lowgrade_by_year = []
    deaths_highgrade_by_year = []

    # initialise all counts
    false_positives = 0
    true_positives = 0
    followups = 0
    screens = 0
    cost_screening_total = 0.0
    cost_followup_total = 0.0
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
            # only in screen if not skip_next
            in_screening = in_screening & ~skip_next

            if screening_AI:
                cost_screening_total += np.sum(in_screening) * cost_screen_ai
            else:
                cost_screening_total += np.sum(in_screening) * cost_screen_no_ai

            screens+=np.sum(in_screening)

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
                in_screening_nocancer = in_screening & ~has_cancer
                false_detections = np.random.rand(np.sum(in_screening_nocancer)) > specificity

                false_positives += np.sum(false_detections)
                followups_this_screen = number_detected_this_screen + np.sum(false_detections)

                #where there are false detections and the person is still in_program, skip the next screen
                skip_next = np.zeros(N, dtype=bool)
                if det_skip_cect:
                    skip_next[in_screening_nocancer] = false_detections

                detected[detected_cancer_indices] = True
                tumor_size[detected_cancer_indices] = 0.0
                # remove from program
                in_program[detected_cancer_indices] = False

                #triaging skip logic
                if triage_skip:
                    #for those with no cancer, skip next according to triage_specificity
                    skip_next[in_screening_nocancer] = np.random.rand(np.sum(in_screening_nocancer)) < triage_specificity

                    # go through small, medium, large and decide who to skip
                    very_small_undetected_indices = np.where(in_screening_nocancer & ~skip_next & (tumor_size <= very_small_threshold))[0]
                    small_undetected_indices = np.where(in_screening_nocancer & ~skip_next & (tumor_size > very_small_threshold) & (tumor_size <= large_threshold))[0]
                    large_undetected_indices = np.where(in_screening_nocancer & ~skip_next & (tumor_size > large_threshold))[0]

                    #skip the next screen for a proportion of these
                    skip_next[very_small_undetected_indices] = np.random.rand(len(very_small_undetected_indices)) < triage_skip_very_small
                    skip_next[small_undetected_indices] = np.random.rand(len(small_undetected_indices)) < triage_skip_small
                    skip_next[large_undetected_indices] = np.random.rand(len(large_undetected_indices)) < triage_skip_large
            else:
                followups_this_screen = 0
                number_detected_this_screen = 0

            # Follow-up cost
            if followups_this_screen > 0:
                if onsite_followup:
                    followups += followups_this_screen
                    cost_followup_total += followups_this_screen * cost_followup_ai
                    # use second specificity to assess who must be sent for second review
                    # all positives after CECT AI must be checked by a radiologist
                    n_second_reviews = np.sum(np.random.rand(np.sum(false_detections)) > second_specificity)
                    # add true positives
                    n_second_reviews +=number_detected_this_screen
                    cost_followup_total += n_second_reviews * cost_second_review_cect
                else:
                    if rad_followup:
                        cost_screening_total += followups_this_screen * cost_followup_rad_ncct # cost of screening follow-up with radiologist on NCCT

                        n_second_reviews = np.sum(np.random.rand(np.sum(false_detections)) > second_specificity)
                        n_second_reviews += number_detected_this_screen
                        followups += n_second_reviews
                        cost_followup_total += n_second_reviews * cost_followup_no_ai # cost of CECT follow-up in hospital outpatients
                    else:
                        followups += followups_this_screen
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
                  )



    results = {
        'total_cost': total_cost,
        'followups': followups,
        'screens': screens,
        'highgrade_detected': total_highgrade_detected,
        'lowgrade_detected': total_lowgrade_detected,
        'cost_screening': cost_screening_total,
        'cost_followup': cost_followup_total,
        'hg_costs': highgrade_costs,
        'lowgrade_costs': lowgrade_costs,
    }
    return results


def run_multiple_simulations_with_sensitivity(n=100,csv_save_path=None):
    no_screen_results = []
    screen_no_ai_results = []
    screen_ai_results = []
    screen_ai2_results = []
    screen_ai2_extra_results = []
    screen_ai2_withskip_results = []
    screen_ai2_extra_withskip_results = []
    screen_ai_radfu_results = []
    screen_ai_triage_results = []
    screen_ai_triage_radfu_results = []
    screen_ai_triage_radfu_extrasens_results = []
    screen_ai_2stage_triage_extrasens_results = []
    screen_ai_2stage_withskip_triaging_results = []
    screen_ai_2stage_withskip_triaging_extraspecific_results = []
    screen_ai_1stage_skiptriage_radfu_extraspecific_results = []
    screen_ai_2stage_withskip_triaging_extraspecific_extrasens_results = []

    for i in range(n):
        print(f"Running simulation {i + 1}/{n}")
        varied_params = vary_all_params(BASE_PARAMS, NO_VARIATION_KEYS)
        seed = i

        #CONFIG INFO:
        #withskip - skip the next round of screening for patients who received a clear CECT from previous screening round
        #radfu - use a single-stage AI on NCCT to filter out scans before radiologist review
        #extrasens - use a more sensitive AI that matches radiologist sensitivity for very small masses
        #triage_skip - use a triage system to skip the next screen for some patients if they fall below the AI's
        #              high-sensitivity threshold

        no_screen = run_simulation(varied_params, screening_on=False, screening_AI=False, seed=seed)
        screen_no_ai = run_simulation(varied_params, screening_on=True, screening_AI=False, seed=seed)
        screen_ai = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1, seed=seed)
        screen_ai2 = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, seed=seed)
        screen_ai2_extra = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, extrasens=True, seed=seed)
        screen_ai2_withskip = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, extrasens=False, with_skip=True, seed=seed)
        screen_ai2_extra_withskip = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                             extrasens=True, with_skip=True, seed=seed)
        screen_ai_radfu = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1, rad_followup=True, seed=seed)
        screen_ai_triage = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1,
                                         rad_followup=False, seed=seed,triage_skip=True)
        screen_ai_triage_radfu = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1,
                                         rad_followup=True, seed=seed,triage_skip=True)
        screen_ai_triage_radfu_extrasens = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1,
                                            rad_followup=True, seed=seed,triage_skip=True,extrasens=True,with_skip=False)
        screen_ai_2stage_triage_extrasens = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                            rad_followup=False, seed=seed,triage_skip=True,extrasens=True,with_skip=False)
        screen_ai_2stage_withskip_triage = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                            rad_followup=False, seed=seed,triage_skip=True,extrasens=False,with_skip=True)
        screen_ai_2stage_withskip_triaging_extraspec = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                            rad_followup=False, seed=seed,triage_skip=True,extrasens=False,with_skip=True,extraspec=True)
        screen_ai_1stage_skiptriage_radfu_extraspec = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1,
                                            rad_followup=True, seed=seed,triage_skip=True,extrasens=False,with_skip=True,extraspec=True)
        screen_ai_2stage_withskip_triaging_extraspecific_extrasens = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                            rad_followup=False, seed=seed,triage_skip=True,extrasens=True,with_skip=True,extraspec=True)
        print()

        no_screen_results.append(no_screen)
        screen_no_ai_results.append(screen_no_ai)
        screen_ai_results.append(screen_ai)
        screen_ai2_results.append(screen_ai2)
        screen_ai2_extra_results.append(screen_ai2_extra)
        screen_ai2_withskip_results.append(screen_ai2_withskip)
        screen_ai2_extra_withskip_results.append(screen_ai2_extra_withskip)
        screen_ai_radfu_results.append(screen_ai_radfu)
        screen_ai_triage_results.append(screen_ai_triage)
        screen_ai_triage_radfu_results.append(screen_ai_triage_radfu)
        screen_ai_triage_radfu_extrasens_results.append(screen_ai_triage_radfu_extrasens)
        screen_ai_2stage_triage_extrasens_results.append(screen_ai_2stage_triage_extrasens)
        screen_ai_2stage_withskip_triaging_results.append(screen_ai_2stage_withskip_triage)
        screen_ai_2stage_withskip_triaging_extraspecific_results.append(screen_ai_2stage_withskip_triaging_extraspec)
        screen_ai_1stage_skiptriage_radfu_extraspecific_results.append(screen_ai_1stage_skiptriage_radfu_extraspec)
        screen_ai_2stage_withskip_triaging_extraspecific_extrasens_results.append(screen_ai_2stage_withskip_triaging_extraspecific_extrasens)


    no_screen_stats = scenario_stats_notreat(no_screen_results)
    screen_no_ai_stats = scenario_stats_notreat(screen_no_ai_results)
    screen_ai_stats = scenario_stats_notreat(screen_ai_results)
    screen_ai2_stats = scenario_stats_notreat(screen_ai2_results)
    screen_ai2_extra_stats = scenario_stats_notreat(screen_ai2_extra_results)
    screen_ai2_withskip_stats = scenario_stats_notreat(screen_ai2_withskip_results)
    screen_ai2_extra_withskip_stats = scenario_stats_notreat(screen_ai2_extra_withskip_results)
    screen_ai_radfu_stats = scenario_stats_notreat(screen_ai_radfu_results)
    screen_ai_triage_stats = scenario_stats_notreat(screen_ai_triage_results)
    screen_ai_triage_radfu_stats = scenario_stats_notreat(screen_ai_triage_radfu_results)
    screen_ai_triage_radfu_extrasens_stats = scenario_stats_notreat(screen_ai_triage_radfu_extrasens_results)
    screen_ai_2stage_triage_extrasens_stats = scenario_stats_notreat(screen_ai_2stage_triage_extrasens_results)
    screen_ai_2stage_withskip_triage_stats = scenario_stats_notreat(screen_ai_2stage_withskip_triaging_results)
    screen_ai_2stage_withskip_triage_extraspec_stats = scenario_stats_notreat(screen_ai_2stage_withskip_triaging_extraspecific_results)
    screen_ai_1stage_skiptriage_radfu_extraspecific_stats = scenario_stats_notreat(screen_ai_1stage_skiptriage_radfu_extraspecific_results)
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats = scenario_stats_notreat(screen_ai_2stage_withskip_triaging_extraspecific_extrasens_results)
    lifeyears_saved_by_noprogression = BASE_PARAMS["lifeyears_saved_by_noprogress"]

    #find the differences of highgrade_detected in each screening arm with no screening, call it A1, A2, A3 etc.
    #find the differences of lowgrade_detected in each screening arm with no screening, call it B1, B2, B3 etc.

    A1 = no_screen_stats['highgrade_detected'][0] -  screen_no_ai_stats['highgrade_detected'][0]
    A2 = no_screen_stats['highgrade_detected'][0] - screen_ai_stats['highgrade_detected'][0]
    A3 = no_screen_stats['highgrade_detected'][0] - screen_ai2_stats['highgrade_detected'][0]
    A4 = no_screen_stats['highgrade_detected'][0] - screen_ai2_extra_stats['highgrade_detected'][0]
    A5 = no_screen_stats['highgrade_detected'][0] - screen_ai2_withskip_stats['highgrade_detected'][0]
    A6 = no_screen_stats['highgrade_detected'][0] - screen_ai2_extra_withskip_stats['highgrade_detected'][0]
    A7 = no_screen_stats['highgrade_detected'][0] - screen_ai_radfu_stats['highgrade_detected'][0]
    A8 = no_screen_stats['highgrade_detected'][0] - screen_ai_triage_stats['highgrade_detected'][0]
    A9 = no_screen_stats['highgrade_detected'][0] - screen_ai_triage_radfu_stats['highgrade_detected'][0]
    A10 = no_screen_stats['highgrade_detected'][0] - screen_ai_triage_radfu_extrasens_stats['highgrade_detected'][0]
    A11 = no_screen_stats['highgrade_detected'][0] - screen_ai_2stage_triage_extrasens_stats['highgrade_detected'][0]
    A12 = no_screen_stats['highgrade_detected'][0] - screen_ai_2stage_withskip_triage_stats['highgrade_detected'][0]
    A13 = no_screen_stats['highgrade_detected'][0] - screen_ai_2stage_withskip_triage_extraspec_stats['highgrade_detected'][0]
    A14 = no_screen_stats['highgrade_detected'][0] - screen_ai_1stage_skiptriage_radfu_extraspecific_stats['highgrade_detected'][0]
    A15 = no_screen_stats['highgrade_detected'][0] - screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['highgrade_detected'][0]

    B1 = screen_no_ai_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B2 = screen_ai_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B3 = screen_ai2_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B4 = screen_ai2_extra_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B5 = screen_ai2_withskip_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B6 = screen_ai2_extra_withskip_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B7 = screen_ai_radfu_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B8 = screen_ai_triage_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B9 = screen_ai_triage_radfu_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B10 = screen_ai_triage_radfu_extrasens_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B11 = screen_ai_2stage_triage_extrasens_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B12 = screen_ai_2stage_withskip_triage_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B13 = screen_ai_2stage_withskip_triage_extraspec_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B14 = screen_ai_1stage_skiptriage_radfu_extraspecific_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]
    B15 = screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['lowgrade_detected'][0] - no_screen_stats['lowgrade_detected'][0]

    # find average of each and report as 'Progressions Prevented'
    no_screen_stats['Progressions Prevented'] = 0
    screen_no_ai_stats['Progressions Prevented'] = (A1 + B1)/2
    screen_ai_stats['Progressions Prevented'] = (A2 + B2)/2
    screen_ai2_stats['Progressions Prevented'] = (A3 + B3)/2
    screen_ai2_extra_stats['Progressions Prevented'] = (A4 + B4)/2
    screen_ai2_withskip_stats['Progressions Prevented'] = (A5 + B5)/2
    screen_ai2_extra_withskip_stats['Progressions Prevented'] = (A6 + B6)/2
    screen_ai_radfu_stats['Progressions Prevented'] = (A7 + B7)/2
    screen_ai_triage_stats['Progressions Prevented'] = (A8 + B8)/2
    screen_ai_triage_radfu_stats['Progressions Prevented'] = (A9 + B9)/2
    screen_ai_triage_radfu_extrasens_stats['Progressions Prevented'] = (A10 + B10)/2
    screen_ai_2stage_triage_extrasens_stats['Progressions Prevented'] = (A11 + B11)/2
    screen_ai_2stage_withskip_triage_stats['Progressions Prevented'] = (A12 + B12)/2
    screen_ai_2stage_withskip_triage_extraspec_stats['Progressions Prevented'] = (A13 + B13)/2
    screen_ai_1stage_skiptriage_radfu_extraspecific_stats['Progressions Prevented'] = (A14 + B14)/2
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['Progressions Prevented'] = (A15 + B15)/2

    no_screen_stats['Cost-Effectiveness'] = 0
    screen_no_ai_stats['Cost-Effectiveness'] = (screen_no_ai_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_no_ai_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_stats['Cost-Effectiveness'] = (screen_ai_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai2_stats['Cost-Effectiveness'] = (screen_ai2_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai2_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai2_extra_stats['Cost-Effectiveness'] = (screen_ai2_extra_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai2_extra_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai2_withskip_stats['Cost-Effectiveness'] = (screen_ai2_withskip_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai2_withskip_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai2_extra_withskip_stats['Cost-Effectiveness'] = (screen_ai2_extra_withskip_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai2_extra_withskip_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_radfu_stats['Cost-Effectiveness'] = (screen_ai_radfu_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_radfu_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_triage_stats['Cost-Effectiveness'] = (screen_ai_triage_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_triage_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_triage_radfu_stats['Cost-Effectiveness'] = (screen_ai_triage_radfu_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_triage_radfu_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_triage_radfu_extrasens_stats['Cost-Effectiveness'] = (screen_ai_triage_radfu_extrasens_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_triage_radfu_extrasens_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_2stage_triage_extrasens_stats['Cost-Effectiveness'] = (screen_ai_2stage_triage_extrasens_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_2stage_triage_extrasens_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_2stage_withskip_triage_stats['Cost-Effectiveness'] = (screen_ai_2stage_withskip_triage_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_2stage_withskip_triage_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_2stage_withskip_triage_extraspec_stats['Cost-Effectiveness'] = (screen_ai_2stage_withskip_triage_extraspec_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_2stage_withskip_triage_extraspec_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_1stage_skiptriage_radfu_extraspecific_stats['Cost-Effectiveness'] = (screen_ai_1stage_skiptriage_radfu_extraspecific_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_1stage_skiptriage_radfu_extraspecific_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['Cost-Effectiveness'] = (screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['total_cost'][0] - no_screen_stats['total_cost'][0]) / (screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['Progressions Prevented']*lifeyears_saved_by_noprogression)

    no_screen_stats['Mean Dose'] = (4*no_screen_stats['number_ldcts'][0] + 10*no_screen_stats['number_cects'][0])/1e5
    screen_no_ai_stats['Mean Dose'] = (4*screen_no_ai_stats['number_ldcts'][0] + 10*screen_no_ai_stats['number_cects'][0])/1e5
    screen_ai_stats['Mean Dose'] = (4*screen_ai_stats['number_ldcts'][0] + 10*screen_ai_stats['number_cects'][0])/1e5
    screen_ai2_stats['Mean Dose'] = (4*screen_ai2_stats['number_ldcts'][0] + 10*screen_ai2_stats['number_cects'][0])/1e5
    screen_ai2_extra_stats['Mean Dose'] = (4*screen_ai2_extra_stats['number_ldcts'][0] + 10*screen_ai2_extra_stats['number_cects'][0])/1e5
    screen_ai2_withskip_stats['Mean Dose'] = (4*screen_ai2_withskip_stats['number_ldcts'][0] + 10*screen_ai2_withskip_stats['number_cects'][0])/1e5
    screen_ai2_extra_withskip_stats['Mean Dose'] = (4*screen_ai2_extra_withskip_stats['number_ldcts'][0] + 10*screen_ai2_extra_withskip_stats['number_cects'][0])/1e5
    screen_ai_radfu_stats['Mean Dose'] = (4*screen_ai_radfu_stats['number_ldcts'][0] + 10*screen_ai_radfu_stats['number_cects'][0])/1e5
    screen_ai_triage_stats['Mean Dose'] = (4*screen_ai_triage_stats['number_ldcts'][0] + 10*screen_ai_triage_stats['number_cects'][0])/1e5
    screen_ai_triage_radfu_stats['Mean Dose'] = (4*screen_ai_triage_radfu_stats['number_ldcts'][0] + 10*screen_ai_triage_radfu_stats['number_cects'][0])/1e5
    screen_ai_triage_radfu_extrasens_stats['Mean Dose'] = (4*screen_ai_triage_radfu_extrasens_stats['number_ldcts'][0] + 10*screen_ai_triage_radfu_extrasens_stats['number_cects'][0])/1e5
    screen_ai_2stage_triage_extrasens_stats['Mean Dose'] = (4*screen_ai_2stage_triage_extrasens_stats['number_ldcts'][0] + 10*screen_ai_2stage_triage_extrasens_stats['number_cects'][0])/1e5
    screen_ai_2stage_withskip_triage_stats['Mean Dose'] = (4*screen_ai_2stage_withskip_triage_stats['number_ldcts'][0] + 10*screen_ai_2stage_withskip_triage_stats['number_cects'][0])/1e5
    screen_ai_2stage_withskip_triage_extraspec_stats['Mean Dose'] = (4*screen_ai_2stage_withskip_triage_extraspec_stats['number_ldcts'][0] + 10*screen_ai_2stage_withskip_triage_extraspec_stats['number_cects'][0])/1e5
    screen_ai_1stage_skiptriage_radfu_extraspecific_stats['Mean Dose'] = (4*screen_ai_1stage_skiptriage_radfu_extraspecific_stats['number_ldcts'][0] + 10*screen_ai_1stage_skiptriage_radfu_extraspecific_stats['number_cects'][0])/1e5
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['Mean Dose'] = (4*screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['number_ldcts'][0] + 10*screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['number_cects'][0])/1e5

    # save these results as rows in a csv file
    if csv_save_path:
        with open(csv_save_path, 'w', newline='') as csvfile:
            fieldnames = ['Scenario', 'total_cost',
                          'number_cects', 'number_ldcts',
                          'highgrade_detected', 'lowgrade_detected', 'cost_screening', 'cost_followup',
                          'Progressions Prevented', 'Cost-Effectiveness', 'Mean Dose']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Scenario': 'No Screening', **no_screen_stats})
            writer.writerow({'Scenario': 'Screening (No AI)', **screen_no_ai_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage', **screen_ai_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage', **screen_ai2_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage Extra Sensitive', **screen_ai2_extra_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Skip', **screen_ai2_withskip_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage Extra Sensitive with Skip', **screen_ai2_extra_withskip_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Rad Follow-up', **screen_ai_radfu_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Triage Skip', **screen_ai_triage_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Triage Skip and Rad Follow-up', **screen_ai_triage_radfu_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Triage Skip, Rad Follow-up, and Extra Sensitive', **screen_ai_triage_radfu_extrasens_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Rad Follow-up and Extra Sensitive', **screen_ai_2stage_triage_extrasens_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Skip and Triage', **screen_ai_2stage_withskip_triage_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage and Triage, Extra Specificity', **screen_ai_2stage_withskip_triage_extraspec_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Skip Triage, Rad Follow-up, Extra Specificity', **screen_ai_1stage_skiptriage_radfu_extraspecific_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Skip Triage, Extra Specificity, Extra Sensitivity', **screen_ai_2stage_withskip_triage_extraspec_extrasens_stats})

    print(f"Averaged over {n} simulations with ±20% parameter variation (allowed params only):")
    print_stats_notreat("No Screening", no_screen_stats)
    print_stats_notreat("Screening (No AI)", screen_no_ai_stats)
    print_stats_notreat("Screening (AI) 1 Stage", screen_ai_stats)
    print_stats_notreat("Screening (AI) 2 Stage", screen_ai2_stats)
    print_stats_notreat("Screening (AI) 2 Stage Extra Sensitive", screen_ai2_extra_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip", screen_ai2_withskip_stats)
    print_stats_notreat("Screening (AI) 2 Stage Extra Sensitive with Skip", screen_ai2_extra_withskip_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Rad Follow-up", screen_ai_radfu_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Triage Skip", screen_ai_triage_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Triage Skip and Rad Follow-up", screen_ai_triage_radfu_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Triage Skip, Rad Follow-up, and Extra Sensitive", screen_ai_triage_radfu_extrasens_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Rad Follow-up and Extra Sensitive", screen_ai_2stage_triage_extrasens_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip and Triage", screen_ai_2stage_withskip_triage_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip and Triage, Extra Specificity", screen_ai_2stage_withskip_triage_extraspec_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Skip Triage, Rad Follow-up, Extra Specificity", screen_ai_1stage_skiptriage_radfu_extraspecific_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip Triage, Extra Specificity, Extra Sensitivity", screen_ai_2stage_withskip_triage_extraspec_extrasens_stats)



run_multiple_simulations_with_sensitivity(n=10,csv_save_path='/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/internal_distribution/supplementary_materials/microsimulation_result.csv')
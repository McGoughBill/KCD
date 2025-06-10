import sys

import numpy as np

from utils import *
import csv
# -----------------------
# Base Global Parameters
# -----------------------

# 0	0.0040645
# 1	0.0002335
# 2	0.0001575
# 3	0.000102
# 4	0.000076
# 5	0.0000785
# 6	0.0000705
# 7	0.0000655
# 8	0.000071
# 9	0.0000675
# 10	0.000063
# 11	0.000084
# 12	0.000085
# 13	0.0001075
# 14	0.000112
# 15	0.000157
# 16	0.000173
# 17	0.000236
# 18	0.0003045
# 19	0.0003465
# 20	0.000345
# 21	0.0003535
# 22	0.0003555
# 23	0.0003955
# 24	0.000377
# 25	0.0003775
# 26	0.000446
# 27	0.0004505
# 28	0.0004805
# 29	0.0005315
# 30	0.0005375
# 31	0.0006125
# 32	0.00065
# 33	0.0007075
# 34	0.00075
# 35	0.000838
# 36	0.00087
# 37	0.000983
# 38	0.0010935
# 39	0.00121
# 40	0.0012865
# 41	0.0014055
# 42	0.0015005
# 43	0.001584
# 44	0.0017405
# 45	0.001889
# 46	0.00204
# 47	0.002316
# 48	0.0024875
# 49	0.0027175
# 50	0.0029745
# 51	0.0031805
# 52	0.003435
# 53	0.0037055
# 54	0.00398
# 55	0.0043125
# 56	0.004698
# 57	0.00512
# 58	0.005497
# 59	0.0058765
# 60	0.0065125
# 61	0.007114
# 62	0.0078255
# 63	0.0085385
# 64	0.0094005
# 65	0.010182
# 66	0.011246
# 67	0.012341
# 68	0.013627
# 69	0.0146985
# 70	0.016238
# 71	0.017641
# 72	0.019433
# 73	0.021059
# 74	0.0229095
# 75	0.0256765
# 76	0.0288005
# 77	0.0324395
# 78	0.0364115
# 79	0.040597
# 80	0.046667
# 81	0.0523075
# 82	0.059117
# 83	0.0654345
# 84	0.073535
# 85	0.0827835
# 86	0.0938435
# 87	0.105431
# 88	0.1201465
# 89	0.134145
# 90	0.151925
# 91	0.1696015
# 92	0.1898345
# 93	0.209864
# 94	0.23207
# 95	0.2513495
# 96	0.274501
# 97	0.302701
# 98	0.3204345
# 99	0.3496805
# 100	0.3782555
# 101	0.395511222
# 102	0.418864789
# 103	0.442218356
# 104	0.465571922
# 105	0.488925489
# 106	0.512279056
# 107	0.535632622
# 108	0.558986189
# 109	0.582339756
# 110	0.605693322
# 111	0.629046889
# 112	0.652400456
# 113	0.675754022
# 114	0.699107589
# 115	0.722461156
# 116	0.745814722
# 117	0.769168289
# 118	0.792521856
# 119	0.815875422

mort_other_dict = {0:0.004065, 1:0.000234, 2:0.000158, 3:0.000102, 4:0.000076, 5:0.000079, 6:0.000071, 7:0.000066, 8:0.000071, 9:0.000068, 10:0.000063, 11:0.000084, 12:0.000085, 13:0.000108, 14:0.000112, 15:0.000157, 16:0.000173, 17:0.000236, 18:0.000305, 19:0.000347, 20:0.000345, 21:0.000354, 22:0.000356, 23:0.000396, 24:0.000377, 25:0.000378, 26:0.000446, 27:0.000451, 28:0.000481, 29:0.000532, 30:0.000538, 31:0.000613, 32:0.00065, 33:0.000708, 34:0.00075, 35:0.000838, 36:0.00087, 37:0.000983, 38:0.001094, 39:0.00121, 40:0.001287, 41:0.001406, 42:0.001501, 43:0.001584, 44:0.001741, 45:0.001889, 46:0.00204, 47:0.002316, 48:0.002488, 49:0.002718, 50:0.002975, 51:0.003181, 52:0.003435, 53:0.003706, 54:0.00398, 55:0.004313, 56:0.004698, 57:0.00512, 58:0.005497, 59:0.005877, 60:0.006513, 61:0.007114, 62:0.007826, 63:0.008539,
                     64:0.009401, 65:0.010182, 66:0.011246, 67:0.012341, 68:0.013627, 69:0.014698, 70:0.016238, 71:0.017641, 72:0.019433, 73:0.021059, 74:0.02291, 75:0.025677, 76:0.028801, 77:0.03244, 78:0.036412, 79:0.040597, 80:0.046667, 81:0.052308, 82:0.059117, 83:0.065435, 84:0.073535, 85:0.082784, 86:0.093844, 87:0.105431, 88:0.120147, 89:0.134145, 90:0.151925, 91:0.169602, 92:0.189835, 93:0.209864, 94:0.23207, 95:0.25135, 96:0.274501, 97:0.302701, 98:0.320435, 99:0.349681, 100:0.378256, 101:0.395511, 102:0.418865, 103:0.442218, 104:0.465572, 105:0.488925, 106:0.512279, 107:0.535633, 108:0.558986, 109:0.58234, 110:0.605693, 111:0.629047, 112:0.6524, 113:0.675754, 114:0.699108, 115:0.722461, 116:0.745815, 117:0.769168, 118:0.792522, 119:0.815875,
                   120:1,121:1,122:1,123:1,124:1}  # age-standardised rate of death: https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/mortalityratesqxbysingleyearofage

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
    "initial_prevalence": 0.005,  # 0.3% tumour prevalence on first screen - taken from YKST (0.25%) and increased 20% to account for missed cases by radiologists
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
    "threshold_7cm": 179594,
    # survival by size https://pmc.ncbi.nlm.nih.gov/articles/PMC2703466/ - perfect paper for what we want to do
    "age40to49_lt4_survival5yr:": 0.955,
    "age50to59_lt4_survival5yr:": 0.942,
    "age60to69_lt4_survival5yr:": 0.942,
    "age70to79_lt4_survival5yr:": 0.966,

    "age40to49_4to7_survival5yr:": 0.955,
    "age50to59_4to7_survival5yr:": 0.912,
    "age60to69_4to7_survival5yr:": 0.903,
    "age70to79_4to7_survival5yr:": 0.966,

    "age40to49_gt7_survival5yr:": 0.886,
    "age50to59_gt7_survival5yr:": 0.836,
    "age60to69_gt7_survival5yr:": 0.865,
    "age70to79_gt7_survival5yr:": 0.838,


    # Growth parameters (no variation) - average of 3 sources below (all between 1.5-2 years)
    "doubling_time_mean": 1.58,  # https://pmc.ncbi.nlm.nih.gov/articles/PMC3989814/, https://onlinelibrary.wiley.com/doi/10.1155/2015/598134, https://academic.oup.com/jjco/article/34/2/82/822167
    "doubling_time_sd": 0.5,

    # Costs (no variation)
    "cost_screen_ai": 20,  # assume overhead, contrast, etc. is absorbed. Use same as Sheffield for unit incremental cost of abd acquistion
    # + £5 for AI
    "cost_screen_no_ai": 30,  # cost_screen_ai + £15 for radiologist
    "cost_followup_ai": 105,  # cost_followup_no_ai  - cost_second_review_cect for unreported CT cost + AI unit cost
    "cost_followup_no_ai": 150,  # no AI follow-up cost, therefore assumed in hospital outpatients
    "cost_MDT": 230,  # multidisciplinary team cost
    "cost_second_review_cect": 50,  # cost of second review of CECT-AI positive Rosenberg, Christian, et al. "Teleradiology from the provider’s perspective—Cost analysis for a mid-size university hospital." European radiology 23 (2013): 2197-2205.
    "cost_followup_rad_ncct": 15,  # ncct_reporting_cost = cost_screen_no_ai - (cost_screen_ai - 5 from no AI)


    # Annual detection probabilities (allowed to vary)
    "annual_detection_prob_lt3":0.03,# Figure 5 in https://www.sciencedirect.com/science/article/pii/S1098301519321370#fig2
    "annual_detection_prob_bt3and7":0.07, # Figure 5 in https://www.sciencedirect.com/science/article/pii/S1098301519321370#fig2
    "annual_detection_prob_gt7":0.29, # Figure 5 in https://www.sciencedirect.com/science/article/pii/S1098301519321370#fig2



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
    "prob_highgrade_at_4cm", "prob_highgrade_at_5cm", "prob_highgrade_at_over5cm","cost_followup_rad_ncct",
    "age40to49_lt4_survival5yr:", "age50to59_lt4_survival5yr:", "age60to69_lt4_survival5yr:", "age70to79_lt4_survival5yr:",
    "age40to49_4to7_survival5yr:", "age50to59_4to7_survival5yr:", "age60to69_4to7_survival5yr:", "age70to79_4to7_survival5yr:",
    "age40to49_gt7_survival5yr:", "age50to59_gt7_survival5yr:", "age60to69_gt7_survival5yr:", "age70to79_gt7_survival5yr:",
    "annual_detection_prob_lt3", "annual_detection_prob_bt3and7", "annual_detection_prob_gt7", "dose_screen", "dose_followup",
}

def assign_grade(sizes, threshold1,threshold2):
    """
    Assigns a grade to each tumour based on its size
    """
    grade = np.zeros(len(sizes))
    grade[(sizes < threshold1) & (sizes>0)] = 1
    grade[(sizes >= threshold1) & (sizes < threshold2)] = 2
    grade[(sizes >= threshold2)] = 3
    return grade



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
    # threshold_1cm = params["threshold_1cm"]
    # threshold_2cm = params["threshold_2cm"]
    threshold_3cm = params["threshold_3cm"]
    threshold_4cm = params["threshold_4cm"]
    # threshold_5cm = params["threshold_5cm"]
    # threshold_6cm = params["threshold_6cm"]
    threshold_7cm = params["threshold_7cm"]
    doubling_time_mean = params["doubling_time_mean"]
    doubling_time_sd = params["doubling_time_sd"]
    cost_screen_ai = params["cost_screen_ai"]
    cost_screen_no_ai = params["cost_screen_no_ai"]
    cost_followup_ai = params["cost_followup_ai"]
    cost_followup_rad_ncct = params["cost_followup_rad_ncct"]
    cost_followup_no_ai = params["cost_followup_no_ai"]
    cost_second_review_cect = params["cost_second_review_cect"]
    cost_MDT = params["cost_MDT"]
    annual_detection_prob_lt3 = params["annual_detection_prob_lt3"]
    annual_detection_prob_bt3and7 = params["annual_detection_prob_bt3and7"]
    annual_detection_prob_gt7 = params["annual_detection_prob_gt7"]
    INITIAL_TUMOUR_SIZE = params["INITIAL_TUMOUR_SIZE"]

    age40to49_lt4_survival5yr = params["age40to49_lt4_survival5yr:"]
    age50to59_lt4_survival5yr = params["age50to59_lt4_survival5yr:"]
    age60to69_lt4_survival5yr = params["age60to69_lt4_survival5yr:"]
    age70to79_lt4_survival5yr = params["age70to79_lt4_survival5yr:"]
    age40to49_4to7_survival5yr = params["age40to49_4to7_survival5yr:"]
    age50to59_4to7_survival5yr = params["age50to59_4to7_survival5yr:"]
    age60to69_4to7_survival5yr = params["age60to69_4to7_survival5yr:"]
    age70to79_4to7_survival5yr = params["age70to79_4to7_survival5yr:"]
    age40to49_gt7_survival5yr = params["age40to49_gt7_survival5yr:"]
    age50to59_gt7_survival5yr = params["age50to59_gt7_survival5yr:"]
    age60to69_gt7_survival5yr = params["age60to69_gt7_survival5yr:"]
    age70to79_gt7_survival5yr = params["age70to79_gt7_survival5yr:"]



    # initialise tracking arrays
    screen_years = set(range(0, years + 1, screen_interval))
    in_program = np.ones(N, dtype=bool)
    age_at_diagnosis = np.full(N, np.nan)
    size_at_diagnosis = np.full(N, np.nan)
    has_cancer = np.zeros(N, dtype=bool)
    tumor_size = np.zeros(N, dtype=float)
    detected = np.zeros(N, dtype=bool)
    grade = np.zeros(N, dtype=bool)
    tumour_start_time = np.full(N, np.nan)
    tumour_growth_factor = np.ones(N)
    mortality_status = np.vectorize(mort_other_dict.get)(ages)
    tumour_detected_time = np.full(N, np.nan)  # the simulation year in which cancer is first detected
    highgrade_at_detection = np.zeros(N, dtype=bool)
    gt7_at_detection = np.zeros(N, dtype=bool)
    bt4_7_at_detection = np.zeros(N, dtype=bool)
    lt_4_at_detection = np.zeros(N, dtype=bool)
    year_of_death = np.full(N, np.nan)
    skip_next = np.zeros(N, dtype=bool)

    # initialise all counts
    false_positives = 0
    true_positives = 0
    followups = 0
    screens = 0
    cost_screening_total = 0.0
    cost_followup_total = 0.0
    # highgrade_costs = 0.0
    # lowgrade_costs = 0.0
    # total_highgrade_detected = 0
    # total_lowgrade_detected = 0
    gt7_costs = 0.0
    bt4_7_costs = 0.0
    lt4_costs = 0.0
    total_gt7_detected = 0
    total_bt4_7_detected = 0
    total_lt4_detected = 0

    # initialise starting cancers
    initial_cancer_indices = np.where((ages >= start_screening_age) & (np.random.rand(N) < initial_prevalence))[0]
    num_initial_cancers = len(initial_cancer_indices)
    if num_initial_cancers > 0:
        initial_sizes = np.random.normal(loc=mean_size, scale=std_size, size=num_initial_cancers*200)
        # large is where volume exceeds threshold_3cm
        is_large = initial_sizes > threshold_3cm
        # resample initial_sizes until we have  57% large and 43% small. do this with replacement
        large_cancers = int(pct_large*num_initial_cancers)
        small_cancers = num_initial_cancers - large_cancers
        large_indices = np.random.choice(np.where(is_large)[0], large_cancers, replace=True)
        small_indices = np.random.choice(np.where(~is_large)[0], small_cancers, replace=True)
        initial_sizes = np.concatenate([initial_sizes[large_indices], initial_sizes[small_indices]])

        # assign the initial sizes to the initial cancers
        tumor_size[initial_cancer_indices] = initial_sizes

        has_cancer[initial_cancer_indices] = True
        tumour_growth_factor = assign_exponential_growth(initial_cancer_indices, doubling_time_mean, doubling_time_sd,
                                                         tumour_growth_factor)
        # change grade to the following based on volume: 0 meaning no cancer, 1 meaning cancer less than 4, 2 to mean between 4 and 7, 3 to mean over 7
        grade[initial_cancer_indices] = assign_grade(initial_sizes, threshold_4cm, threshold_7cm)

    # main simulation loop
    for year in range(0, years + 1):
        program_indices = np.where(in_program)[0]
        # highgrade_newdetected_count = 0
        # lowgrade_newdetected_count = 0
        gt7_newdetected_count = 0
        bt4_7_newdetected_count = 0
        lt4_newdetected_count = 0


        # Growth
        cancer_indices = np.where(in_program & has_cancer)[0]  # Corrected
        if len(cancer_indices) > 0:
            prev_sizes = tumor_size[cancer_indices].copy()
            tumor_size[cancer_indices] *= tumour_growth_factor[cancer_indices]
            grade = assign_grade(tumor_size, threshold_4cm, threshold_7cm)

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
                age_at_diagnosis[detected_cancer_indices] = ages[detected_cancer_indices]
                size_at_diagnosis[detected_cancer_indices] = sizes_current[detection_array]
                # highgrade_at_detection[detected_cancer_indices] = highgrade[detected_cancer_indices]
                gt7_at_detection[detected_cancer_indices] = sizes_current[detection_array] > threshold_7cm
                bt4_7_at_detection[detected_cancer_indices] = (sizes_current[detection_array] > threshold_4cm) & (sizes_current[detection_array] <= threshold_7cm)
                lt_4_at_detection[detected_cancer_indices] = sizes_current[detection_array] <= threshold_4cm

                number_detected_this_screen = np.sum(detection_array)
                gt7_detected_count = np.sum(gt7_at_detection[detected_cancer_indices])
                bt4_7_detected_count = np.sum(bt4_7_at_detection[detected_cancer_indices])
                lt4_detected_count = np.sum(lt_4_at_detection[detected_cancer_indices])
                total_gt7_detected += gt7_detected_count
                total_bt4_7_detected += bt4_7_detected_count
                total_lt4_detected += lt4_detected_count

                gt7_newdetected_count += gt7_detected_count
                bt4_7_newdetected_count += bt4_7_detected_count
                lt4_newdetected_count += lt4_detected_count

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
            # lowgrade_costs += lowgrade_newdetected_count * cost_MDT
            # highgrade_costs += highgrade_newdetected_count * cost_MDT
            gt7_costs += gt7_newdetected_count * cost_MDT
            bt4_7_costs += bt4_7_newdetected_count * cost_MDT
            lt4_costs += lt4_newdetected_count * cost_MDT

        # Annual detection for all non-screened cancers - serendipitous detection
        undetected_cancer_indices = np.where(in_program & ~detected & has_cancer)[0]
        if len(undetected_cancer_indices) > 0:
            gt7_undetected_indices = undetected_cancer_indices[tumor_size[undetected_cancer_indices] > threshold_7cm]
            bt3and7_undetected_indices = undetected_cancer_indices[(tumor_size[undetected_cancer_indices] > threshold_3cm) & (tumor_size[undetected_cancer_indices] <= threshold_7cm)]
            lt3_undetected_indices = undetected_cancer_indices[tumor_size[undetected_cancer_indices] <= threshold_3cm]

            detected_small_indices = lt3_undetected_indices[np.random.rand(len(lt3_undetected_indices)) < annual_detection_prob_lt3] if len(lt3_undetected_indices) > 0 else np.array([], dtype=int)
            detected_medium_indices = bt3and7_undetected_indices[np.random.rand(len(bt3and7_undetected_indices)) < annual_detection_prob_bt3and7] if len(bt3and7_undetected_indices) > 0 else np.array([], dtype=int)
            detected_large_indices = gt7_undetected_indices[np.random.rand(len(gt7_undetected_indices)) < annual_detection_prob_gt7] if len(gt7_undetected_indices) > 0 else np.array([], dtype=int)

            detected_annual_indices = np.concatenate((detected_small_indices, detected_medium_indices, detected_large_indices))

            if len(detected_annual_indices) > 0:
                number_detected_annual = len(detected_annual_indices)
                detected[detected_annual_indices] = True
                age_at_diagnosis[detected_annual_indices] = ages[detected_annual_indices]
                size_at_diagnosis[detected_annual_indices] = tumor_size[detected_annual_indices]
                # total_highgrade_detected += np.sum(highgrade[detected_annual_indices])
                # total_lowgrade_detected += number_detected_annual - np.sum(highgrade[detected_annual_indices])
                # lowgrade_newdetected_count += number_detected_annual - np.sum(highgrade[detected_annual_indices])
                # highgrade_newdetected_count += np.sum(highgrade[detected_annual_indices])
                # highgrade_at_detection[detected_annual_indices] = highgrade[detected_annual_indices]
                # tumour_detected_time[detected_annual_indices] = year
                # true_positives += np.sum(highgrade[detected_annual_indices]) + np.sum(
                #     ~highgrade[detected_annual_indices])
                total_gt7_detected += np.sum(tumor_size[detected_annual_indices] > threshold_7cm)
                total_bt4_7_detected += np.sum((tumor_size[detected_annual_indices] > threshold_4cm) & (tumor_size[detected_annual_indices] <= threshold_7cm))
                total_lt4_detected += np.sum(tumor_size[detected_annual_indices] <= threshold_3cm)
                gt7_newdetected_count += np.sum(tumor_size[detected_annual_indices] > threshold_7cm)
                bt4_7_newdetected_count += np.sum((tumor_size[detected_annual_indices] > threshold_4cm) & (tumor_size[detected_annual_indices] <= threshold_7cm))
                lt4_newdetected_count += np.sum(tumor_size[detected_annual_indices] <= threshold_3cm)
                tumour_detected_time[detected_annual_indices] = year
                true_positives += number_detected_annual
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
                grade[newly_cancerous] = 1
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

    end_sizes = size_at_diagnosis[~np.isnan(size_at_diagnosis)]
    end_ages = age_at_diagnosis[~np.isnan(age_at_diagnosis)]
    total_lifeyears_following_detection = 0
    for size, age in zip(end_sizes, end_ages):
        if size<=threshold_4cm:
            if age>=40 and age<50:
                total_lifeyears_following_detection += age40to49_lt4_survival5yr*5
            elif age>=50 and age<60:
                total_lifeyears_following_detection += age50to59_lt4_survival5yr*5
            elif age>=60 and age<70:
                total_lifeyears_following_detection += age60to69_lt4_survival5yr*5
            elif age>=70 and age<80:
                total_lifeyears_following_detection += age70to79_lt4_survival5yr*5
        elif size>threshold_4cm and size<=threshold_7cm:
            if age>=40 and age<50:
                total_lifeyears_following_detection += age40to49_4to7_survival5yr*5
            elif age>=50 and age<60:
                total_lifeyears_following_detection += age50to59_4to7_survival5yr*5
            elif age>=60 and age<70:
                total_lifeyears_following_detection += age60to69_4to7_survival5yr*5
            elif age>=70 and age<80:
                total_lifeyears_following_detection += age70to79_4to7_survival5yr*5
        elif size>threshold_7cm:
            if age>=40 and age<50:
                total_lifeyears_following_detection += age40to49_gt7_survival5yr*5
            elif age>=50 and age<60:
                total_lifeyears_following_detection += age50to59_gt7_survival5yr*5
            elif age>=60 and age<70:
                total_lifeyears_following_detection += age60to69_gt7_survival5yr*5
            elif age>=70 and age<80:
                total_lifeyears_following_detection += age70to79_gt7_survival5yr*5

    undetected_lt4 = np.sum((tumor_size <= threshold_4cm) & ~detected & has_cancer)
    undetected_bt4_7 = np.sum((tumor_size > threshold_4cm) & (tumor_size <= threshold_7cm) & ~detected & has_cancer)
    undetected_gt7 = np.sum((tumor_size > threshold_7cm) & ~detected & has_cancer)

    total_lifeyears_undetected_screen_end = 0
    sizes_undetected = tumor_size[~detected & has_cancer]
    ages_undetected = ages[~detected & has_cancer]

    for size, age in zip(sizes_undetected, ages_undetected):
        if size<=threshold_4cm:
            if age>=40 and age<50:
                total_lifeyears_undetected_screen_end += age40to49_lt4_survival5yr*5
            elif age>=50 and age<60:
                total_lifeyears_undetected_screen_end += age50to59_lt4_survival5yr*5
            elif age>=60 and age<70:
                total_lifeyears_undetected_screen_end += age60to69_lt4_survival5yr*5
            elif age>=70 and age<80:
                total_lifeyears_undetected_screen_end += age70to79_lt4_survival5yr*5
        elif size>threshold_4cm and size<=threshold_7cm:
            if age>=40 and age<50:
                total_lifeyears_undetected_screen_end += age40to49_4to7_survival5yr*5
            elif age>=50 and age<60:
                total_lifeyears_undetected_screen_end += age50to59_4to7_survival5yr*5
            elif age>=60 and age<70:
                total_lifeyears_undetected_screen_end += age60to69_4to7_survival5yr*5
            elif age>=70 and age<80:
                total_lifeyears_undetected_screen_end += age70to79_4to7_survival5yr*5
        elif size>threshold_7cm:
            if age>=40 and age<50:
                total_lifeyears_undetected_screen_end += age40to49_gt7_survival5yr*5
            elif age>=50 and age<60:
                total_lifeyears_undetected_screen_end += age50to59_gt7_survival5yr*5
            elif age>=60 and age<70:
                total_lifeyears_undetected_screen_end += age60to69_gt7_survival5yr*5
            elif age>=70 and age<80:
                total_lifeyears_undetected_screen_end += age70to79_gt7_survival5yr*5

    total_lifeyears_following_screening = total_lifeyears_following_detection + total_lifeyears_undetected_screen_end


    results = {
        'total_cost': total_cost,
        'followups': followups,
        'screens': screens,
        'gt7_detected': total_gt7_detected,
        'bt4_7_detected': total_bt4_7_detected,
        'lt4_detected': total_lt4_detected,
        'cost_screening': cost_screening_total,
        'cost_followup': cost_followup_total,
        'gt7_costs': gt7_costs,
        'bt4_7_costs': bt4_7_costs,
        'lt4_costs': lt4_costs,
        'total_lifeyears_following_screening': total_lifeyears_following_screening,
        'undetected_lt4': undetected_lt4,
        'undetected_bt4_7': undetected_bt4_7,
        'undetected_gt7': undetected_gt7
    }
    return results


def run_multiple_simulations_with_sensitivity(n=100,csv_save_path=None, screen_interval=2,
                                              initial_prevalence=0.003):
    no_screen_results = []
    screen_no_ai_results = []
    screen_ai_results = []
    screen_ai2_results = []
    screen_ai2_withskip_results = []
    screen_ai_radfu_results = []
    screen_ai_triage_results = []
    screen_ai_triage_radfu_results = []
    screen_ai_2stage_withskip_triaging_results = []
    screen_ai_2stage_withskip_triaging_extraspecific_extrasens_results = []

    for i in range(n):
        print(f"Running simulation {i + 1}/{n}")
        varied_params = vary_all_params(BASE_PARAMS, NO_VARIATION_KEYS)
        varied_params['screen_interval'] = screen_interval
        varied_params['initial_prevalence'] = initial_prevalence
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
        screen_ai2_withskip = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2, extrasens=False, with_skip=True, seed=seed)
        screen_ai_radfu = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1, rad_followup=True, seed=seed)
        screen_ai_triage = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1,
                                         rad_followup=False, seed=seed,triage_skip=True)
        screen_ai_triage_radfu = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=1,
                                         rad_followup=True, seed=seed,triage_skip=True)
        screen_ai_2stage_withskip_triage = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                            rad_followup=False, seed=seed,triage_skip=True,extrasens=False,with_skip=True)
        screen_ai_2stage_withskip_triaging_extraspecific_extrasens = run_simulation(varied_params, screening_on=True, screening_AI=True, ai_stages=2,
                                            rad_followup=False, seed=seed,triage_skip=True,extrasens=True,with_skip=True,extraspec=True)
        print()

        no_screen_results.append(no_screen)
        screen_no_ai_results.append(screen_no_ai)
        screen_ai_results.append(screen_ai)
        screen_ai2_results.append(screen_ai2)
        screen_ai2_withskip_results.append(screen_ai2_withskip)
        screen_ai_radfu_results.append(screen_ai_radfu)
        screen_ai_triage_results.append(screen_ai_triage)
        screen_ai_triage_radfu_results.append(screen_ai_triage_radfu)
        screen_ai_2stage_withskip_triaging_results.append(screen_ai_2stage_withskip_triage)
        screen_ai_2stage_withskip_triaging_extraspecific_extrasens_results.append(screen_ai_2stage_withskip_triaging_extraspecific_extrasens)


    no_screen_stats = scenario_stats_notreat(no_screen_results)
    screen_no_ai_stats = scenario_stats_notreat(screen_no_ai_results)
    screen_ai_stats = scenario_stats_notreat(screen_ai_results)
    screen_ai2_stats = scenario_stats_notreat(screen_ai2_results)
    screen_ai2_withskip_stats = scenario_stats_notreat(screen_ai2_withskip_results)
    screen_ai_radfu_stats = scenario_stats_notreat(screen_ai_radfu_results)
    screen_ai_triage_stats = scenario_stats_notreat(screen_ai_triage_results)
    screen_ai_triage_radfu_stats = scenario_stats_notreat(screen_ai_triage_radfu_results)
    screen_ai_2stage_withskip_triage_stats = scenario_stats_notreat(screen_ai_2stage_withskip_triaging_results)
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats = scenario_stats_notreat(screen_ai_2stage_withskip_triaging_extraspecific_extrasens_results)

    no_screen_stats['Mean Dose'] = (4*no_screen_stats['number_ldcts'][0] + 10*no_screen_stats['number_cects'][0])/1e5
    screen_no_ai_stats['Mean Dose'] = (4*screen_no_ai_stats['number_ldcts'][0] + 10*screen_no_ai_stats['number_cects'][0])/1e5
    screen_ai_stats['Mean Dose'] = (4*screen_ai_stats['number_ldcts'][0] + 10*screen_ai_stats['number_cects'][0])/1e5
    screen_ai2_stats['Mean Dose'] = (4*screen_ai2_stats['number_ldcts'][0] + 10*screen_ai2_stats['number_cects'][0])/1e5
    screen_ai2_withskip_stats['Mean Dose'] = (4*screen_ai2_withskip_stats['number_ldcts'][0] + 10*screen_ai2_withskip_stats['number_cects'][0])/1e5
    screen_ai_radfu_stats['Mean Dose'] = (4*screen_ai_radfu_stats['number_ldcts'][0] + 10*screen_ai_radfu_stats['number_cects'][0])/1e5
    screen_ai_triage_stats['Mean Dose'] = (4*screen_ai_triage_stats['number_ldcts'][0] + 10*screen_ai_triage_stats['number_cects'][0])/1e5
    screen_ai_triage_radfu_stats['Mean Dose'] = (4*screen_ai_triage_radfu_stats['number_ldcts'][0] + 10*screen_ai_triage_radfu_stats['number_cects'][0])/1e5
    screen_ai_2stage_withskip_triage_stats['Mean Dose'] = (4*screen_ai_2stage_withskip_triage_stats['number_ldcts'][0] + 10*screen_ai_2stage_withskip_triage_stats['number_cects'][0])/1e5
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['Mean Dose'] = (4*screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['number_ldcts'][0] + 10*screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['number_cects'][0])/1e5


    no_screen_stats['small_table'] = True
    screen_no_ai_stats['small_table'] = True
    screen_ai_stats['small_table'] = False
    screen_ai2_stats['small_table'] = False
    screen_ai2_withskip_stats['small_table'] = False
    screen_ai_radfu_stats['small_table'] = False
    screen_ai_triage_stats['small_table'] = False
    screen_ai_triage_radfu_stats['small_table'] = True
    screen_ai_2stage_withskip_triage_stats['small_table'] = True
    screen_ai_2stage_withskip_triage_extraspec_extrasens_stats['small_table'] = True

    # save these results as rows in a csv file
    if csv_save_path:
        with open(csv_save_path, 'w', newline='') as csvfile:
            fieldnames = ['Scenario', 'total_cost',
                          'number_cects', 'number_ldcts',
                          'gt7_detected', 'bt4_7_detected', 'lt4_detected',
                          'cost_screening', 'cost_followup','total_lifeyears_following_screening',
                            'undetected_lt4', 'undetected_bt4_7', 'undetected_gt7',
                           'Mean Dose','small_table']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Scenario': 'No Screening', **no_screen_stats})
            writer.writerow({'Scenario': 'Screening (No AI)', **screen_no_ai_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage', **screen_ai_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage', **screen_ai2_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Skip', **screen_ai2_withskip_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Rad Follow-up', **screen_ai_radfu_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Triage Skip', **screen_ai_triage_stats})
            writer.writerow({'Scenario': 'Screening (AI) 1 Stage with Triage Skip and Rad Follow-up', **screen_ai_triage_radfu_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Skip and Triage', **screen_ai_2stage_withskip_triage_stats})
            writer.writerow({'Scenario': 'Screening (AI) 2 Stage with Skip Triage, Extra Specificity, Extra Sensitivity', **screen_ai_2stage_withskip_triage_extraspec_extrasens_stats})

    print(f"Averaged over {n} simulations with ±20% parameter variation (allowed params only):")
    print_stats_notreat("No Screening", no_screen_stats)
    print_stats_notreat("Screening (No AI)", screen_no_ai_stats)
    print_stats_notreat("Screening (AI) 1 Stage", screen_ai_stats)
    print_stats_notreat("Screening (AI) 2 Stage", screen_ai2_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip", screen_ai2_withskip_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Rad Follow-up", screen_ai_radfu_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Triage Skip", screen_ai_triage_stats)
    print_stats_notreat("Screening (AI) 1 Stage with Triage Skip and Rad Follow-up", screen_ai_triage_radfu_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip and Triage", screen_ai_2stage_withskip_triage_stats)
    print_stats_notreat("Screening (AI) 2 Stage with Skip Triage, Extra Specificity, Extra Sensitivity", screen_ai_2stage_withskip_triage_extraspec_extrasens_stats)



# params = [[2,0.003],
#           [2,0.005],
#           [4,0.003],
# run_multiple_simulations_with_sensitivity(n=1000, screen_interval=61, initial_prevalence=0.005,
#                                           csv_save_path=f'/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/internal_distribution/supplementary_materials/microsimulation_result_{61}_{0.0005}_noscreenvalid.csv')
# run_multiple_simulations_with_sensitivity(n=100000, screen_interval=2, initial_prevalence=0.005,
#                                           csv_save_path=f'/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/internal_distribution/supplementary_materials/microsimulation_result_{2}_{0.0005}_ykstvalid.csv')
params = [  [1,0.005], [2,0.005],      [4,0.005],[10,0.005],[21,0.005]]
# params = [   [1,0.005]]
#
for intv,prev in params:
    print(f"Running simulation with interval {intv} and prevalence {prev}")
    run_multiple_simulations_with_sensitivity(n=1000, screen_interval=intv, initial_prevalence=prev,
                                              csv_save_path=f'/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/internal_distribution/supplementary_materials/microsimulation_result_{intv}_{prev}.csv')
    print()
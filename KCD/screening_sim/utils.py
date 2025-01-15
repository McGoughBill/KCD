import numpy as np
import math

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

def vary_all_params(base_params, NO_VARIATION_KEYS):
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

def assign_exponential_growth(indices, doubling_time_mean, doubling_time_sd, tumour_growth_factor):
    if len(indices) == 0:
        return
    dt = np.random.normal(doubling_time_mean, doubling_time_sd, size=len(indices))
    dt = np.where(dt <= 0.3, 0.3, dt)
    growth_factors = 2 ** (1.0 / dt)
    tumour_growth_factor[indices] = growth_factors
    return tumour_growth_factor

def update_hgnancy_on_size_crossing(prev_sizes, new_sizes, idx, highgrade, threshold_2cm, threshold_4cm, prob_highgrade_at_2cm, prob_highgrade_at_4cm):
    not_hg = ~highgrade[idx]
    if np.any(not_hg):
        idx_nm = idx[not_hg]
        prev = prev_sizes[not_hg]
        new = new_sizes[not_hg]

        crossed_2cm = (prev <= threshold_2cm) & (new > threshold_2cm)
        if np.any(crossed_2cm):
            c2_indices = idx_nm[crossed_2cm]
            turn_hg_2 = np.random.rand(len(c2_indices)) < prob_highgrade_at_2cm
            highgrade[c2_indices[turn_hg_2]] = True

        still_not_hg = idx_nm[~highgrade[idx_nm]]
        if len(still_not_hg) > 0:
            still_prev = prev[~highgrade[idx_nm]]
            still_new = new[~highgrade[idx_nm]]
            crossed_4cm = (still_prev <= threshold_4cm) & (still_new > threshold_4cm)
            if np.any(crossed_4cm):
                c4_indices = still_not_hg[crossed_4cm]
                turn_hg_4 = np.random.rand(len(c4_indices)) < prob_highgrade_at_4cm
                highgrade[c4_indices[turn_hg_4]] = True

    return highgrade

def update_hgnancy_on_size_crossing_multithresh(
    prev_sizes,
    new_sizes,
    idx,
    highgrade,
    thresholds=None,
    probs=None
):
    """
    Update hgnancy status for lumps based on whether they've crossed
    any of the specified thresholds between old and new size measurements.

    :param prev_sizes: array-like of previous sizes
    :param new_sizes:  array-like of new sizes
    :param idx:        indices of lumps to consider (e.g., np.arange(len(prev_sizes)))
    :param highgrade:  boolean array: True if lump is highgrade, False otherwise
    :param thresholds: list or array of thresholds to check (e.g. [1,2,3,4,5])
    :param probs:      list or array of probabilities for each threshold (same length as thresholds)
    :return:           updated 'highgrade' array
    """

    # Default thresholds and probabilities for crossing each centimeter 1–5
    if thresholds is None:
        thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
    if probs is None:
        # Example probabilities – adjust to your needs
        probs = [0.05, 0.1, 0.15, 0.25, 0.4]

    # Focus on those lumps that are not yet highgrade
    not_hg = ~highgrade[idx]
    if np.any(not_hg):
        idx_nm = idx[not_hg]
        prev = prev_sizes[not_hg]
        new = new_sizes[not_hg]

        # Check sequentially for crossing each threshold
        for threshold, p_hg in zip(thresholds, probs):
            crossed = (prev <= threshold) & (new > threshold)
            if np.any(crossed):
                # For lumps that crossed this threshold, flip to highgrade by probability
                crossing_indices = idx_nm[crossed]
                turns_highgrade = np.random.rand(len(crossing_indices)) < p_hg
                highgrade[crossing_indices[turns_highgrade]] = True

            # Remove any lumps that have already turned highgrade from further checks
            still_not_hg = ~highgrade[idx_nm]
            idx_nm = idx_nm[still_not_hg]
            prev = prev[still_not_hg]
            new = new[still_not_hg]

    return highgrade

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

def scenario_stats(results):
    m_deaths_other, s_deaths_other = avg_std_metric(results, 'deaths_other')
    m_deaths_lowgrade, s_deaths_lowgrade = avg_std_metric(results, 'deaths_lowgrade')
    m_deaths_highgrade, s_deaths_highgrade = avg_std_metric(results, 'deaths_highgrade')

    m_lowgrade_detected, s_lowgrade_detected = avg_std_metric(results, 'lowgrade_detected')
    m_highgrade_detected, s_highgrade_detected = avg_std_metric(results, 'highgrade_detected')

    m_cost_treatment, s_cost_treatment = avg_std_metric(results, 'cost_treatment')

    m_cost_screening, s_cost_screening = avg_std_metric(results, 'cost_screening')
    m_cost_followup, s_cost_followup = avg_std_metric(results, 'cost_followup')
    m_cost_detection_noscreen, s_cost_detection_noscreen = avg_std_metric(results, 'cost_detection_noscreen')

    m_total_cost, s_total_cost = avg_std_metric(results, 'total_cost')
    m_total_ly, s_total_ly = avg_std_metric(results, 'total_life_years')

    m_number_followups, s_number_followups = avg_std_metric(results, 'followups')

    m_hg_costs, s_hg_costs = avg_std_metric(results, 'hg_costs')
    m_lowgrade_costs, s_lowgrade_costs = avg_std_metric(results, 'lowgrade_costs')

    avg_hg_cost = m_hg_costs / m_highgrade_detected if m_highgrade_detected > 0 else 0
    avg_lowgrade_cost = m_lowgrade_costs / m_lowgrade_detected if m_lowgrade_detected > 0 else 0

    # 5-year survival
    m_5yr_lowgrade, s_5yr_lowgrade = avg_std_metric(results, '5yr_survival_lowgrade')
    m_5yr_hg, s_5yr_hg = avg_std_metric(results, '5yr_survival_highgrade')

    return {
        'deaths_other': (m_deaths_other, s_deaths_other),
        'deaths_lowgrade': (m_deaths_lowgrade, s_deaths_lowgrade),
        'deaths_highgrade': (m_deaths_highgrade, s_deaths_highgrade),
        'lowgrade_detected': (m_lowgrade_detected, s_lowgrade_detected),
        'highgrade_detected': (m_highgrade_detected, s_highgrade_detected),
        'cost_treatment': (m_cost_treatment, s_cost_treatment),
        'cost_screening': (m_cost_screening, s_cost_screening),
        'cost_followup': (m_cost_followup, s_cost_followup),
        'number_followups': (m_number_followups, s_number_followups),
        'cost_detection_noscreen': (m_cost_detection_noscreen, s_cost_detection_noscreen),
        'total_cost': (m_total_cost, s_total_cost),
        'avg_hg_cost': avg_hg_cost,
        'avg_lowgrade_cost': avg_lowgrade_cost,
        'total_life_years': (m_total_ly, s_total_ly),
        # 5-year survival means
        '5yr_survival_lowgrade': (m_5yr_lowgrade, s_5yr_lowgrade),
        '5yr_survival_highgrade': (m_5yr_hg, s_5yr_hg),
    }

def scenario_stats_notreat(results):
    m_lowgrade_detected, s_lowgrade_detected = avg_std_metric(results, 'lowgrade_detected')
    m_highgrade_detected, s_highgrade_detected = avg_std_metric(results, 'highgrade_detected')
    m_cost_screening, s_cost_screening = avg_std_metric(results, 'cost_screening')
    m_cost_followup, s_cost_followup = avg_std_metric(results, 'cost_followup')
    m_cost_detection_noscreen, s_cost_detection_noscreen = avg_std_metric(results, 'cost_detection_noscreen')
    m_total_cost, s_total_cost = avg_std_metric(results, 'total_cost')
    m_number_followups, s_number_followups = avg_std_metric(results, 'followups')

    return {
        'lowgrade_detected': (m_lowgrade_detected, s_lowgrade_detected),
        'highgrade_detected': (m_highgrade_detected, s_highgrade_detected),
        'cost_screening': (m_cost_screening, s_cost_screening),
        'cost_followup': (m_cost_followup, s_cost_followup),
        'cost_detection_noscreen': (m_cost_detection_noscreen, s_cost_detection_noscreen),
        'total_cost': (m_total_cost, s_total_cost),
        'number_followups': (m_number_followups, s_number_followups),
    }

def avg_std_metric(results_list, metric):
    values = [r[metric] for r in results_list]
    return np.mean(values), np.std(values)

def fmt(mean, std):
    # Format with 3 sig figs and scale
    return f"{pretty_number(mean)} ({pretty_number(std)})"

def print_stats(name, stats):
    print(f"{name} results:")
    print(f"  Total cost: £{fmt(stats['total_cost'][0], stats['total_cost'][1])}")
    print(f"  Total life years: {fmt(stats['total_life_years'][0], stats['total_life_years'][1])}")
    print(f"  Deaths (other): {fmt(stats['deaths_other'][0], stats['deaths_other'][1])}")
    print(f"  Deaths (lowgrade): {fmt(stats['deaths_lowgrade'][0], stats['deaths_lowgrade'][1])}")
    print(f"  Deaths (highgrade): {fmt(stats['deaths_highgrade'][0], stats['deaths_highgrade'][1])}")
    print(f"  lowgrade detected: {fmt(stats['lowgrade_detected'][0], stats['lowgrade_detected'][1])}")
    print(f"  highgrade detected: {fmt(stats['highgrade_detected'][0], stats['highgrade_detected'][1])}")
    print(f"  Cost of treatment: £{fmt(stats['cost_treatment'][0], stats['cost_treatment'][1])}")
    print(f"  Cost of screening: £{fmt(stats['cost_screening'][0], stats['cost_screening'][1])}")
    print(f"  Cost of follow-up: £{fmt(stats['cost_followup'][0], stats['cost_followup'][1])}")
    print(f"  5-year survival (lowgrade): {fmt(stats['5yr_survival_lowgrade'][0], stats['5yr_survival_lowgrade'][1])}")
    print(f"  5-year survival (highgrade): {fmt(stats['5yr_survival_highgrade'][0], stats['5yr_survival_highgrade'][1])}")
    print(f"  Average cost of highgrade detection: £{fmt(stats['avg_hg_cost'], 0)}")
    print(f"  Average cost of lowgrade detection: £{fmt(stats['avg_lowgrade_cost'], 0)}")
    print(f"  Number of follow-ups: {fmt(stats['number_followups'][0], stats['number_followups'][1])}")
    print(f"  Cost of detection (no screen): £{fmt(stats['cost_detection_noscreen'][0], stats['cost_detection_noscreen'][1])}")

def print_stats_notreat(name, stats):
    print(f"{name} results:")
    print(f"  Total cost: £{fmt(stats['total_cost'][0], stats['total_cost'][1])}")
    print(f"  lowgrade detected: {fmt(stats['lowgrade_detected'][0], stats['lowgrade_detected'][1])}")
    print(f"  highgrade detected: {fmt(stats['highgrade_detected'][0], stats['highgrade_detected'][1])}")
    print(f"  Cost of screening: £{fmt(stats['cost_screening'][0], stats['cost_screening'][1])}")
    print(f"  Cost of follow-up: £{fmt(stats['cost_followup'][0], stats['cost_followup'][1])}")
    print(f"  Number of follow-ups: {fmt(stats['number_followups'][0], stats['number_followups'][1])}")
    print(f"  Cost of detection (no screen): £{fmt(stats['cost_detection_noscreen'][0], stats['cost_detection_noscreen'][1])}")
    print(f"  Progressions Prevented: {stats['Progressions Prevented']:.2f}")
    print(f"  Cost-Effectiveness: £{stats['Cost-Effectiveness']:.2f}")
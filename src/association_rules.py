from itertools import combinations

import numpy as np
import pandas as pd


def association_rules(df, metric="confidence", min_threshold=0.8, support_only=False):
   
    if df.empty:
        raise ValueError(
            "The input DataFrame `df` containing the frequent itemsets is empty."
        )

    # Check required columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError(
            "DataFrame must contain the columns 'support' and 'itemsets'."
        )

    def conviction_helper(sAC, sA, sC):
        confidence = sAC / sA
        conviction = np.full(confidence.shape, np.inf, dtype=float)
        mask = confidence < 1.0
        conviction[mask] = (1.0 - sC[mask]) / (1.0 - confidence[mask])
        return conviction

    def zhangs_metric_helper(sAC, sA, sC):
        numerator = metric_dict["leverage"](sAC, sA, sC)
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator == 0, 0, numerator / denominator)

    # Metrics dictionary
    metric_dict = {
        "antecedent support": lambda _, sA, __: sA,
        "consequent support": lambda _, __, sC: sC,
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC / sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC) / sC,
        "leverage": lambda sAC, sA, sC: sAC - sA * sC,
        "conviction": conviction_helper,
        "zhangs_metric": zhangs_metric_helper,
    }

    columns_ordered = [
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
    ]

    # Enforce metric if support_only
    if support_only:
        metric = "support"
    elif metric not in metric_dict:
        raise ValueError(f"Metric must be one of {list(metric_dict.keys())}, got '{metric}'")

    # Build frequent itemset support dict
    frequent_items = dict(zip(
        map(frozenset, df["itemsets"].values),
        df["support"].values
    ))

    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # Generate rules
    for itemset in frequent_items:
        sAC = frequent_items[itemset]
        for i in range(len(itemset) - 1, 0, -1):
            for antecedent in combinations(itemset, r=i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent

                if not consequent:
                    continue

                if support_only:
                    sA = sC = None
                else:
                    try:
                        sA = frequent_items[antecedent]
                        sC = frequent_items[consequent]
                    except KeyError as e:
                        raise KeyError(
                            f"{e}\nLikely missing antecedent/consequent support."
                            " Use `support_only=True` to skip computing other metrics."
                        )

                score = metric_dict[metric](sAC, sA, sC)
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + columns_ordered)

    rule_supports = np.array(rule_supports).T.astype(float)
    df_rules = pd.DataFrame({
        "antecedents": rule_antecedents,
        "consequents": rule_consequents
    })

    if support_only:
        df_rules["support"] = rule_supports[0]
        for col in columns_ordered:
            if col != "support":
                df_rules[col] = np.nan
    else:
        sAC, sA, sC = rule_supports
        for col in columns_ordered:
            df_rules[col] = metric_dict[col](sAC, sA, sC)

    return df_rules

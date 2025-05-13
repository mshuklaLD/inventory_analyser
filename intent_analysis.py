# intent_analysis.py

keyword_map = {
    "slowest moving": {
        "metric": "Aging", "agg": "mean", "order": "desc", "group_by": "Category"
    },
    "fastest moving": {
        "metric": "Aging", "agg": "mean", "order": "asc", "group_by": "Category"
    },
    "most number of slow moving": {
        "metric": "Aging", "agg": "count_above", "threshold": 90, "order": "desc", "group_by": "Carat Range"
    },
    "most slow moving goods": {
        "metric": "Aging", "agg": "count_above", "threshold": 90, "order": "desc", "group_by": "Carat Range"
    },
    "highest sales": {
        "metric": "My Sales", "agg": "sum", "order": "desc", "group_by": "Category"
    },
    "least sales": {
        "metric": "My Sales", "agg": "sum", "order": "asc", "group_by": "Category"
    },
    "surplus": {
        "metric": "My Stock", "agg": "sum", "order": "desc", "group_by": "Category"
    }
}


def detect_analysis_intent(user_question):
    question = user_question.lower()
    for key in sorted(keyword_map.keys(), key=len, reverse=True):
        if key in question:
            return keyword_map[key]
    return None


def generate_summary_from_intent(df, config):
    metric = config["metric"]
    group_by = config.get("group_by", "Category")
    order = config["order"]
    agg = config["agg"]

    if group_by not in df.columns or metric not in df.columns:
        return "Required column not found in data."

    try:
        if agg == "mean":
            result = df.groupby(group_by)[metric].mean()
        elif agg == "sum":
            result = df.groupby(group_by)[metric].sum()
        elif agg == "count_above":
            threshold = config.get("threshold", 90)
            df_filtered = df[df[metric] > threshold]
            result = df_filtered.groupby(group_by).size()
        else:
            return "Unsupported aggregation method."

        result_sorted = result.sort_values(ascending=(order == "asc")).head(10)
        return result_sorted.to_string()
    except Exception as e:
        return f"Error during summary generation: {e}"

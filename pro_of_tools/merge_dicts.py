def merge_datasets(datasets_list):
    merged = {}

    for ds in datasets_list:
        for group, curves in ds.items():
            if group not in merged:
                merged[group] = []
            # curves 本身就是 list
            merged[group].extend(curves)

    return merged

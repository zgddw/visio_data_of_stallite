from pro_of_tools import stname
import warnings

def standardize_data(raw_data, alias_map):
    """
    raw_data : dict
        原始数据字典（键名混乱）
    alias_map : dict
        同义词 → 标准名
    """
    std = {}

    for key, value in raw_data.items():
        if key in alias_map:
            std_key = alias_map[key]
            std[std_key] = value
        else:
            # 不认识的量直接忽略（或记录日志）
            pass

    return std


def build_datasets(raw_data):
    data = standardize_data(raw_data, stname.ALIASES)
    t = data.get("t")

    if t is None:
        warnings.warn("时间轴 't' 不存在，返回空 datasets")
        return {}

    plot_vars = {
        "Bt": [("Bt")],
        "Bxyz": [("Bx"), ("By"), ("Bz")],
        "theta": [("theta")],
        "phi": [("phi")],
        "Vp": [("Vp")],
        "Tp": [("Tp")],
        "Np": [("Np")],
        "Ne":[("Ne")],
        "Pdy": [("Pdy")],
        "beta": [("beta")],
        "NevsEnergy":[("NevsEnergy")],
        "SymH":[("SymH")],
        "GIC":[("GIC")]
    }

    datasets = {}

    for group, vars_list in plot_vars.items():
        curves = []

        for var_name in vars_list:
            y = data.get(var_name)
            if y is None:
                continue
            if len(y) != len(t):
                continue

            style_info = stname.PLOT_STYLE.get(var_name, {})
            curves.append({
                "x": t,
                "y": y,
                "label": style_info.get("unit", ""),
                "unit": style_info.get("unit", ""),
                "style": style_info.get("style", {})
            })

        if curves:
            datasets[group] = curves

    return datasets
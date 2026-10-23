ALIASES = {

    # ============================================================
    # 时间（所有任务通用）
    # ============================================================
    "Epoch": "t",
    "EPOCH": "t",
    "epoch": "t",
    "Time": "t",
    "time": "t",
    "datetime": "t",
    "UTC": "t",
    "t":"t",

    # OMNI
    "Epoch_UTC": "t",

    # PSP / SOLO / STEREO
    "SCET": "t",
    "UTC_TIME": "t",

    # ============================================================
    # 磁场总强度 |B|
    # ============================================================
    "B": "Bt",
    "BT": "Bt",
    "Bmag": "Bt",
    "BMAG": "Bt",
    "MAGNITUDE": "Bt",
    "BF1":"Bt",
    "bt":"Bt",
    "B1F1":"Bt",
    # OMNI
    "ABS_B": "Bt",

    # ACE / Wind
    "B_TOTAL": "Bt",

    # PSP
    "B_MAG": "Bt",

    # SOLO
    "B_RTN_MAG": "Bt",

    # STEREO
    "BFIELD": "Bt",

    # ============================================================
    # 磁场分量 Bx
    # ============================================================
    "Bx": "Bx",
    "BX": "Bx",
    "bx":"Bx",
    # Wind / ACE
    "BX_GSE": "Bx",
    "BX_GSM": "Bx",

    # OMNI
    "BX_GSM": "Bx",

    # PSP
    "B_X": "Bx",

    # SOLO
    "B_RTN_X": "Bx",

    # STEREO
    "BX_RTN": "Bx",

    # ============================================================
    # 磁场分量 By
    # ============================================================
    "By": "By",
    "BY": "By",
    "by":"By",
    # Wind / ACE
    "BY_GSE": "By",
    "BY_GSM": "By",

    # OMNI
    "BY_GSM": "By",

    # PSP
    "B_Y": "By",

    # SOLO
    "B_RTN_Y": "By",

    # STEREO
    "BY_RTN": "By",

    # ============================================================
    # 磁场分量 Bz
    # ============================================================
    "Bz": "Bz",
    "BZ": "Bz",
    "bz":"Bz",
    # Wind / ACE
    "BZ_GSE": "Bz",
    "BZ_GSM": "Bz",

    # OMNI
    "BZ_GSM": "Bz",

    # PSP
    "B_Z": "Bz",

    # SOLO
    "B_RTN_Z": "Bz",

    # STEREO
    "BZ_RTN": "Bz",

    # ============================================================
    # 磁场方向角 theta
    # ============================================================
    "theta": "theta",
    "Theta": "theta",
    "THETA": "theta",

    # Wind / ACE
    "THETA_B": "theta",

    # OMNI
    "THETA": "theta",

    # PSP / SOLO
    "B_THETA": "theta",

    # ============================================================
    # 磁场方向角 phi
    # ============================================================
    "phi": "phi",
    "Phi": "phi",
    "PHI": "phi",

    # Wind / ACE
    "PHI_B": "phi",

    # OMNI
    "PHI": "phi",

    # PSP / SOLO
    "B_PHI": "phi",

    # ============================================================
    # 太阳风速度 Vp
    # ============================================================
    "V": "Vp",
    "Vsw": "Vp",
    "Vp": "Vp",
    "SPEED": "Vp",
    "P_VELS":"Vp",
    # OMNI
    "V_SW": "Vp",

    # Wind / ACE
    "V_GSE": "Vp",
    "V_GSM": "Vp",

    # PSP
    "VEL": "Vp",
    "vp_fit_RTN":"Vp",
    "vp_moment_RTN":"Vp",
    # SOLO / STEREO
    "V_RTN": "Vp",

    # ============================================================
    # 质子温度 Tp
    # ============================================================
    "T": "Tp",
    "Tp": "Tp",
    "TEMP": "Tp",

    # OMNI
    "T_P": "Tp",

    # Wind / ACE
    "PROTON_TEMP": "Tp",
    "P_TEMP":"Tp",
    # PSP / SOLO
    "T_PROTON": "Tp",
    "wp_fit":"Tp",
    # ============================================================
    # 质子密度 Np
    # ============================================================
    "N": "Np",
    "Np": "Np",
    "DENSITY": "Np",

    # OMNI
    "N_P": "Np",

    # Wind / ACE
    "PROTON_DENSITY": "Np",
    "P_DENS":"Np",
    # PSP / SOLO / STEREO
    "N_PROTON": "Np",
    "np_fit":"Np",
    "np_moment":"Np",

    # ============================================================
    # 电子密度 Ne
    # ============================================================
    "ne":'Ne',
    "electron_density":"Ne",
    "nevsenergy":"NevsEnergy",
    "EFLUX_VS_ENERGY":"NevsEnergy",
    "elect_density":'Ne',
    "Electron_lo":'Ne',

    # ============================================================
    # 重离子密度
    # ============================================================
    "n_tc_he2plus":"Nhe2",
    "A_He":"Nhe2",
    "n_tc_o6plus":"No6",
    "A_O":"No6",
    "n_tc_fe10plus":"Nfe10",
    "A_Fe":"Nfe10",
    "n_tc_cplus":"Nc",
    "A_C":"Nc",
    "A_He_Flux":"Fluxhe",
    "A_O_Flux":"FluxO",
    "A_Fe_Flux":"Fluxfe",
    "A_C_Flux":"Fluxc",
    # ============================================================
    # 动压 Pdy
    # ============================================================
    "Pdy": "Pdy",
    "PDYN": "Pdy",
    "DYN_P": "Pdy",
    'Pth':'Pdy',
    # OMNI
    "P_DYN": "Pdy",

    # Wind / ACE
    "PRESSURE": "Pdy",

    # ============================================================
    # 等离子体 beta
    # ============================================================
    "beta": "beta",
    "Beta": "beta",
    "BETA": "beta",

    # OMNI
    "BETA_PLASMA": "beta",

    # Wind / ACE
    "PLASMA_BETA": "beta",

    # PSP / SOLO
    "BETA_P": "beta",
    "SymH":"SymH",
    "SYM_H":"SymH",
    "GIC":"GIC"
}
linewith = 2
PLOT_STYLE = {
    "Bt": {
        "unit": f"{r'$B_t$'}\n(nT)",
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Bx": {
        "unit": "nT",
        "style": {"color": "r", "ls": "-", "lw": linewith}
    },
    "By": {
        "unit": "nT",
        "style": {"color": "g", "ls": "-", "lw": linewith}
    },
    "Bz": {
        "unit": "nT",
        "style": {"color": "b", "ls": "-", "lw": linewith}
    },
    "theta": {
        "unit": f'{r"$\theta$"}\n({r"$^\circ$"})',
        "style": {"color": "black", "ls": "-", "lw":linewith}
    },
    "phi": {
        "unit": f'{r"$\phi$"}\n({r"$^\circ$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Vp": {
        "unit": f'{r'$V_p$'}\n(km/s)',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Np": {
        "unit": f'{r'$N_p$'}\n({r"cm$^{-2}$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Tp": {
        "unit": f"{r'$T_p$'}\n(K)",
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Pdy": {
        "unit": f"{r'$P_{dy}$'}\n(nPa)",
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "beta": {
        "unit": f'{r'$\beta$'}\n ',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "SymH":{
        "unit":'SymH\n(nT)',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "GIC":{"unit":'GIC\n(A)',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Ne":{"unit":f'{r'$N_e$'}\n({r"cm$^{-2}$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Nhe2":{"unit":f'{r'$N_{{He}^{2+}}$'}\n({r"cm$^{-2}$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "No6":{"unit":f'{r'$N_{O^{6+}}$'}\n({r"cm$^{-2}$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Nfe10":{"unit":f'{r'$N_{{Fe}^{10+}}$'}\n({r"cm$^{-2}$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Nc":{"unit":f'{r'$N_C$'}\n({r"cm$^{-2}$"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Fluxhe":{"unit":f'{r'$Particle Intensity$'}\n({r"[cm$^{-2}$sr$^{-1}$s$^{-1}$]"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Fluxo":{"unit":f'{r'$Particle Intensity$'}\n({r"[cm$^{-2}$sr$^{-1}$s$^{-1}$]"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Fluxfe":{"unit":f'{r'$Particle Intensity$'}\n({r"[cm$^{-2}$sr$^{-1}$s$^{-1}$]"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "Fluxc":{"unit":f'{r'$Particle Intensity$'}\n({r"[cm$^{-2}$sr$^{-1}$s$^{-1}$]"})',
        "style": {"color": "black", "ls": "-", "lw": linewith}
    },
    "NevsEnergy":{"unit":f'{'Energy'}\n(eV)',
                  "style":{"shading":'auto','cmap':'jet'}
    }
}

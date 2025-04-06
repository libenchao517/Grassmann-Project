################################################################################
## 可视化实验结果
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
## 导入模块
import os
import numpy as np
import pandas as pd
from Draw import Error_Drawing
from Utils import Make_Table
################################################################################
MT = Make_Table()
root = "GUMAP-Results"
IDXS = ["AMI", "ARI", "HMS"]
DATAS = ["ETH-80", "Weizmann", "EYB", "UTD-MHAD", "UCF-S", "UT-Kinect"]

color_dict = {
    "AMI" : ["#F9D5DD", "#E3738B"],
    "ARI" : ["#FFE8CE", "#FCB462"],
    "HMS" : ["#DCE4FA", "#8CA5EA"],
}

for id in IDXS:
    xlsx_path = os.path.join(root, "GUMAP-"+id+".xlsx")
    mean = pd.read_excel(xlsx_path, sheet_name="Mean", index_col=0, header=0)
    std = pd.read_excel(xlsx_path, sheet_name="Std", index_col=0, header=0)
    mean_t = mean.T
    std_t = std.T

    methods = [mean.index[i] for i in range(0, 20, 2)]

    print_methodS = [
        'CGMKE', 'GKM', 'GLBG', 'SDPCG', 'GAF',
        'GAG', 'GDBS', 'GHDBS', 'GOPT', 'GSP']

    res = pd.DataFrame(index=methods, columns=mean.columns)
    for med in methods:
        res.loc[med] = mean.loc["GUMAP-" + med] - mean.loc[med]

    if id == "time":
        res = -res

    res.to_excel(os.path.join("GUMAP-Results", "GUMAP-" + id + "-Enhance.xlsx"))
    MT.Make(os.path.join("GUMAP-Results", "GUMAP-" + id + "-Enhance.xlsx"))

    for dn in DATAS:
        ED = Error_Drawing(
            path="Figure",
            xlabel="Clustering Methods",
            ylabel="Mean " + id,
            fontsize=18,
            titlefontsize=20,
            formats=["png"]
        )
        mv = np.array(mean_t.loc[dn]).reshape((10, 2)).T
        sv = np.array(std_t.loc[dn]).reshape((10, 2)).T
        flag = True if id != "time" else False
        ED.filename = "fig-bar-" + dn + "-" + id
        ED.drawing_bar_error(
            x_value=print_methodS,
            mean_value=mv,
            std_value=sv,
            labels=["MCFG", "MCFG with GUMAP"],
            colors=color_dict.get(id),
            ylim_flag=flag
        )

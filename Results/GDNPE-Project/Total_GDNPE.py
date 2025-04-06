################################################################################
# 本代码用于GDNPE整理实验结果
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入必要模块
import os
import re
import shutil
import numpy as np
import pandas as pd
from Utils import Make_Table
################################################################################
# 定义基本变量
project = "GDNPE"
index = ["ACC", "F1", "PRE", "REC", "time"]
fromdir = os.getcwd()
todir = project + "-Results"
os.makedirs(todir, exist_ok=True)
Results_list = [res for res in os.listdir() if re.match(rf"{project}-2024*", res)]
num_rst = np.array(np.array(Results_list).shape[0])
MT = Make_Table()
################################################################################
# 整理结果
for idx in index:
    for rst in Results_list:
        result_path = os.path.join(rst, "Result-Files", idx + ".xlsx")
        rst_idx = pd.read_excel(result_path, project, header=0, index_col=0)
        rst_idx = rst_idx.to_numpy()
        num_method = rst_idx.shape[0]
        num_data = rst_idx.shape[1]
        break
    break

for idx in index:
    result_total = np.zeros((num_rst, num_method, num_data))
    for i, rst in enumerate(Results_list):
        result_path = os.path.join(rst, "Result-Files", idx + ".xlsx")
        rst_idx = pd.read_excel(result_path, project, header=0, index_col=0)
        result_idx = rst_idx.to_numpy()
        result_total[i, :, :] = result_idx
    result_mean = np.nanmean(result_total, axis=0)
    result_mean = pd.DataFrame(result_mean, index=rst_idx.index, columns=rst_idx.columns)
    result_std = np.nanstd(result_total, axis=0)
    result_std = pd.DataFrame(result_std, index=rst_idx.index, columns=rst_idx.columns)
    with pd.ExcelWriter(project+ '-' +idx + '.xlsx') as writer:
        result_mean.to_excel(writer, sheet_name="Mean")
        result_std.to_excel(writer, sheet_name="Std")

    MT.Make(project+ '-' + idx + '.xlsx')
    shutil.move(fromdir + "/" + project+ '-' + idx + '.xlsx', todir)

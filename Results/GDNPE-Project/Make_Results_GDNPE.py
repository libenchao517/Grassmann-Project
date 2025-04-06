################################################################################
# 本代码用于整理Grassmann DNPE实验结果
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
import shutil
import datetime as dt
import pandas as pd
from Utils import Make_Table
from GDNPE import GrassmannDNPE_config
config = GrassmannDNPE_config()
################################################################################
# 定义必要变量
Root = "/".join(Path(__file__).parts[:-1])
Analysis = '/Analysis/'
Temp_File= '/Temp-Files/'
Result_File= '/Result-Files/'
Path("./Temp-Files").mkdir(exist_ok=True)
Path("./Result-Files").mkdir(exist_ok=True)
index = ["ACC",  "PRE", "F1", "REC",
         "train-size", "sampling", "time"]
GDNPE_method = [
    "GKNN", "GSVM", "GKDA",
    "GLPP", "GALL", "GNPE-I", "GNPE-II",
    "GPML", "GrNet",
    "NG", "SNG", "GRLGQ",
    "GSNPE-I", "GSNPE-II",
    "GDNPE-I", "GDNPE-II"]
GSNPE_method = ["GNPE-I", "GSNPE-I", "GNPE-II", "GSNPE-II"]
MT = Make_Table(methods = GDNPE_method)
################################################################################
# 整理到临时汇总文件
for d in config.GDNPE_data:
    temp = pd.DataFrame()
    path = Root + Analysis + d
    xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
    for xlsx in xlsx_list:
        df = pd.read_excel(xlsx, header=0, index_col=0)
        temp = pd.concat([temp, df])
    xlsx_path = d + '.xlsx'
    temp.to_excel(Root + Temp_File + xlsx_path)
################################################################################
# 汇总相同维度分析的所有结果
Total_Results = pd.DataFrame()
path = Root + Temp_File
xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    df = pd.read_excel(xlsx, header=0, index_col=0)
    Total_Results = pd.concat([Total_Results, df])
################################################################################
# 整理相同维度分析的各个指标
for idx in index:
    Results = Total_Results[['Method', 'Datasets', idx]].copy()
    Results.set_index(['Method', 'Datasets'], inplace=True)
    Result_GDNPE = pd.DataFrame(index=GDNPE_method, columns=config.GDNPE_data)
    Result_GSNPE = pd.DataFrame(index=GSNPE_method, columns=config.GDNPE_data)
    for m in GDNPE_method:
        for d in config.GDNPE_data:
            try:
                Result_GDNPE.loc[m, d] = Results.loc[m, d][0]
            except:
                Result_GDNPE.loc[m, d] = None
    for m in GSNPE_method:
        for d in config.GDNPE_data:
            try:
                Result_GSNPE.loc[m, d] = Results.loc[m, d][0]
            except:
                Result_GSNPE.loc[m, d] = None
    with pd.ExcelWriter(Root + Result_File + idx + '.xlsx') as writer:
        Result_GDNPE.to_excel(writer, sheet_name="GDNPE")
        Result_GSNPE.to_excel(writer, sheet_name="GSNPE")
################################################################################
# 整理文件和文件夹
fromdir = os.getcwd()
todir = "GDNPE-"+str(dt.date.today()) + "-" + dt.datetime.now().time().strftime("%H-%M")
Path("./" + todir).mkdir(exist_ok=True)
shutil.move(fromdir + "/Analysis", todir)
shutil.move(fromdir + "/Result-Files", todir)
shutil.move(fromdir + "/Temp-Files", todir)
shutil.move(fromdir + "/Figure", todir)
if os.path.exists(os.path.join(fromdir, "log_files")):
    shutil.move(os.path.join(fromdir, "log_files"), todir)
################################################################################
# 整理结果文件格式
xlsx_list = list(map(str, list(Path(todir).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    MT.Make(xlsx)

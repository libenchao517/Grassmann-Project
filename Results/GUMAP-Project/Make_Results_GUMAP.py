################################################################################
# 本代码用于整理Grassmann UMAP实验结果
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
from GUMAP import GrassmannUMAP_config
config = GrassmannUMAP_config()
################################################################################
# 定义必要变量
Root = "/".join(Path(__file__).parts[:-1])
Analysis = '/Analysis/'
Temp_File= '/Temp-Files/'
Result_File= '/Result-Files/'
Path("./Temp-Files").mkdir(exist_ok=True)
Path("./Result-Files").mkdir(exist_ok=True)
index = ["ARI", "AMI", "NMI", "HMS", "CMS", "VMS", "FMS", "ACC", "time"]
GUMAP_method = [
    "CGMKE", "GUMAP-CGMKE", "GKM", "GUMAP-GKM",
    "GLBG", "GUMAP-GLBG", "SNNDPCG", "GUMAP-SNNDPCG",
    "GAF", "GUMAP-GAF", "GAG", "GUMAP-GAG",
    "GDBS", "GUMAP-GDBS", "GHDBS",  "GUMAP-GHDBS",
    "GOPT", "GUMAP-GOPT", "GSC", "GUMAP-GSC"]
methods = [
    "CGMKE", "GUMAP-CGMKE", "CGMKE-Enhanced",
    "GKM", "GUMAP-GKM", "GKM-Enhanced",
    "GLBG", "GUMAP-GLBG", "GLBG-Enhanced",
    "SNNDPCG", "GUMAP-SNNDPCG", "SNNDPCG-Enhanced",
    "GAF", "GUMAP-GAF", "GAF-Enhanced",
    "GAG", "GUMAP-GAG", "GAG-Enhanced",
    "GDBS", "GUMAP-GDBS", "GDBS-Enhanced",
    "GHDBS", "GUMAP-GHDBS", "GHDBS-Enhanced",
    "GOPT", "GUMAP-GOPT", "GOPT-Enhanced",
    "GSC", "GUMAP-GSC", "GSC-Enhanced"
]
Enhanced = [item for item in methods if item.endswith("-Enhanced")]
MT = Make_Table(methods = methods)
################################################################################
# 整理到临时汇总文件
for d in config.GUMAP_data:
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
    Result_GUMAP = pd.DataFrame(index=GUMAP_method, columns=config.GUMAP_data)
    Result = pd.DataFrame(index=methods, columns=config.GUMAP_data)
    Result_Enhance = pd.DataFrame(index=Enhanced, columns=config.GUMAP_data)
    for enhance in Enhanced:
        m1 = "-".join(enhance.split("-")[:-1])
        m2 = "GUMAP-" + m1
        for d in config.GUMAP_data:
            try:
                Result.loc[enhance, d] = Results.loc[m2, d][0] - Results.loc[m1, d][0]
                Result_Enhance.loc[enhance, d] = Results.loc[m2, d][0] - Results.loc[m1, d][0]
            except:
                Result.loc[enhance, d] = None
                Result_Enhance.loc[enhance, d] = None
    with pd.ExcelWriter(Root + Result_File + idx + '.xlsx') as writer:
        Result_GUMAP.to_excel(writer, sheet_name="GUMAP")
        Result_Enhance.to_excel(writer, sheet_name="Enhance")
        Result.to_excel(writer, sheet_name="total")
################################################################################
# 整理文件和文件夹
fromdir = os.getcwd()
todir = "GUMAP-"+str(dt.date.today()) + "-" + dt.datetime.now().time().strftime("%H-%M")
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

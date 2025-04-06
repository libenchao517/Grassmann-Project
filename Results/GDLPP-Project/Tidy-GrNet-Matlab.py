################################################################################
## 对比实验12：Matlab上Grassmann Network实验结果整理
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
## 导入模块
from Assess import Analysis_Riemannian
from Factory import factory
from GDLPP import GrassmannDLPP_config
from Utils import Auto_Run
import os
from scipy.io import loadmat
import time
from icecream import ic
################################################################################
## 定义基本变量
config = GrassmannDLPP_config()
################################################################################
## 运行实验
for i in range(1, 11):
    for dn in config.GDLPP_data:
        model = factory(
            func_name='GrNet',
            data_name=dn,
            return_time=config.return_time,
            train_size=config.train_size.get(dn),
            random_state=config.random_state,
            sec_part='Comparatation',
            sec_num=12
        )
        model.Product_Riemannian_DR_Object(
            n_components=config.low_dimensions.get(dn),
            n_neighbors=config.n_neighbors,
            p_grassmann=config.grassmann_p.get(dn)
        )
        pred_path = os.path.join("GDLPP-Results-Matlab/", dn, dn + "-" + str(config.grassmann_p.get(dn)) + "-" + str(i) + "_prelabel.mat")
        real_path = os.path.join("GDLPP-Results-Matlab/", dn, dn + "-" + str(config.grassmann_p.get(dn)) + "-" + str(i) + "_real_label.mat")
############################################################
## 读取结果
        pred = loadmat(pred_path)
        real = loadmat(real_path)
        model.Grassmann_Contrast_Object.t_pred = pred.get("file_prelabel").flatten()
        model.Grassmann_Contrast_Object.target_test = real.get("gr_label").flatten()
        model.Grassmann_Contrast_Object.space = None
        model.Grassmann_Contrast_Object.time = None
        model.Grassmann_Contrast_Object.para[-1] = str(i)
        Analysis_Riemannian(model.Grassmann_Contrast_Object).Analysis()
############################################################
## 整理单次结果
    AR = Auto_Run(
            Project="GDLPP",
            MRPY=None,
            content="Results",
            run_file="Make_Results_GDLPP.py",
            lock=True
    )
    AR.Run()
    time.sleep(100)
############################################################
## 汇总结果
AR = Auto_Run(
    Project="GDLPP",
    MRPY=None,
    content="Results/GDLPP-Project",
    run_file="Total_GDLPP.py",
    lock=True
)
AR.Run()

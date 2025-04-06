################################################################################
## 对比实验13: Deep Network on Grassmann Manifold 效果测试
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
from GDNPE import GrassmannDNPE_config
import os
from scipy.io import loadmat
################################################################################
## 定义基本变量
config = GrassmannDNPE_config()
epoch = 3
################################################################################
## 运行实验
for dn in config.GDNPE_data:
    model = factory(
        func_name='GrNet',
        data_name=dn,
        return_time=config.return_time,
        train_size=config.train_size.get(dn),
        random_state=config.random_state,
        sec_part='Comparatation',
        sec_num=13)
    model.Product_Riemannian_DR_Object(
        n_components=config.low_dimensions.get(dn),
        n_neighbors=config.n_neighbors,
        p_grassmann=config.grassmann_p.get(dn),
        converged_tol=config.converged_tol,
        max_epoch=config.max_epoch
    )
    pred_path = os.path.join("GrNet-pred", dn, dn + "-" + str(epoch) + "-pred" + ".mat")
    sets = loadmat(pred_path)
    model.Grassmann_Contrast_Object.space = None
    model.Grassmann_Contrast_Object.time = None
    model.Grassmann_Contrast_Object.t_pred = sets.get("file_prelabel").flatten()
    model.Grassmann_Contrast_Object.target_test = sets.get("gr_label").flatten()
    Analysis_Riemannian(model.Grassmann_Contrast_Object).Analysis()

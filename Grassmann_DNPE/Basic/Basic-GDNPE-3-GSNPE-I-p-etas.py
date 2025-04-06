################################################################################
## 基础实验3：Grassmann空间的阶数p和能量保留比例eta寻优
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
## 导入模块
import numpy as np
from Assess import Analysis_Riemannian
from Draw import Annotated_Heatmaps
from Draw import Draw_Line_Chart
from Factory import factory
from GDNPE import GrassmannDNPE_config
import gc
################################################################################
## 定义基本变量
config = GrassmannDNPE_config()
max_epoch = 5
p_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
eta_list = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
################################################################################
## 运行实验
for dn in config.basic_data:
    KNN_ = np.zeros((max_epoch, len(eta_list), len(p_list)))
    for epoch in range(max_epoch):
        config.random_state = np.random.randint(2024)
        for i, p in enumerate(p_list):
            for j, eta in enumerate(eta_list):
                model = factory(
                    func_name='GSNPE',
                    data_name=dn,
                    return_time=config.return_time,
                    train_size=config.train_size.get(dn),
                    random_state=config.random_state,
                    sec_part='Basic',
                    sec_num=3)

                model.Product_Grassmann_NPE_Object(
                    n_components=None,
                    n_neighbors=config.n_neighbors,
                    p_grassmann=p,
                    converged_tol=config.converged_tol,
                    max_epoch=config.max_epoch,
                    mode=1,
                    is_discriminant=config.is_discriminant.get(model.func_name),
                    is_semi_supervised=config.is_semi_supervised.get(model.func_name),
                    is_self_supervised=config.is_self_supervised.get(model.func_name))

                model.Grassmann_NPE_Object.n_components = config.low_dimensions.get(dn + "-" + str(p)).get(str(eta))
                model.Grassmann_NPE_Object.fit_transform(model.data, model.target)
                AR = Analysis_Riemannian(model.Grassmann_NPE_Object, save_file=False)
                AR.Analysis()
                KNN_[epoch, j, i] = AR.knn.accuracy

    KNN_mean = np.nanmean(KNN_, axis=0)

    AH = Annotated_Heatmaps(
        filename=model.sec_part + "-" + str(model.sec_num) + "-" + model.func_name + "-" + dn + "-best-p-etas",
        path="./Figure/" + dn,
        fontsize=8,
        xlabel="The Order of Linear Subspaces $p$",
        ylabel="Remaining Energe Ratio " + chr(951),
        xticklabels=p_list,
        yticklabels=eta_list)
    AH.Drawing(KNN_mean)

    Draw_Line_Chart(
        filename=model.sec_part + "-" + str(model.sec_num) + "-" + model.func_name + "-" + dn + "-best-p-with-different-etas",
        path="./Figure/" + dn,
        xlabel="The Order of Linear Subspaces $p$",
        ylabel_left="Mean Classification Accuracy",
        column=p_list,
        left=[KNN_mean[0], KNN_mean[1], KNN_mean[2], KNN_mean[3], KNN_mean[4], KNN_mean[5]],
        ylim_left=(0, 1.05),
        left_marker=("^", "v", "<", ">", "s", "o"),
        left_color=[
            "#427AB2", "#F09148", "#FF9896",
            "#DBDB8D", "#C59D94", "#AFC7E8"],
        left_label=[
            chr(951) + "=0.75", chr(951) + "=0.80",
            chr(951) + "=0.85", chr(951) + "=0.90",
            chr(951) + "=0.95", chr(951) + "=0.99"]).Draw_simple_line()

    gc.collect()

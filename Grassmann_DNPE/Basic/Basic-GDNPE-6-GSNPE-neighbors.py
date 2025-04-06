################################################################################
## 基础实验6：Grassmann Semi-Supervised NPE-I的近邻数寻优
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
from Draw import Draw_Line_Chart
from Draw import Error_Drawing
from Factory import factory
from GDNPE import GrassmannDNPE_config
import gc
################################################################################
## 定义基本变量
config = GrassmannDNPE_config()
max_epoch = 5
n_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
################################################################################
## 运行实验
for dn in config.basic_data:
    KNN_1 = np.zeros((max_epoch, len(n_list)))
    KNN_2 = np.zeros((max_epoch, len(n_list)))
    for epoch in range(max_epoch):
        config.random_state = np.random.randint(2024)
        for i, nn in enumerate(n_list):
            model_SNPE_1 = factory(
                func_name='GSNPE',
                data_name=dn,
                return_time=config.return_time,
                train_size=config.train_size.get(dn),
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=6)

            model_SNPE_2 = factory(
                func_name='GSNPE',
                data_name=dn,
                return_time=config.return_time,
                train_size=config.train_size.get(dn),
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=6)

            model_SNPE_1.Product_Grassmann_NPE_Object(
                n_components=config.low_dimensions.get(dn),
                n_neighbors=nn,
                p_grassmann=config.grassmann_p.get(dn),
                converged_tol=config.converged_tol,
                max_epoch=config.max_epoch,
                mode=1,
                is_discriminant=config.is_discriminant.get(model_SNPE_1.func_name),
                is_semi_supervised=config.is_semi_supervised.get(model_SNPE_1.func_name),
                is_self_supervised=config.is_self_supervised.get(model_SNPE_1.func_name))

            model_SNPE_2.Product_Grassmann_NPE_Object(
                n_components=config.low_dimensions.get(dn),
                n_neighbors=nn,
                p_grassmann=config.grassmann_p.get(dn),
                converged_tol=config.converged_tol,
                max_epoch=config.max_epoch,
                mode=2,
                is_discriminant=config.is_discriminant.get(model_SNPE_2.func_name),
                is_semi_supervised=config.is_semi_supervised.get(model_SNPE_2.func_name),
                is_self_supervised=config.is_self_supervised.get(model_SNPE_2.func_name))

            model_SNPE_1.Grassmann_NPE_Object.fit_transform(model_SNPE_1.data, model_SNPE_1.target)
            AR = Analysis_Riemannian(model_SNPE_1.Grassmann_NPE_Object, save_file=False)
            AR.Analysis()
            KNN_1[epoch, i] = AR.knn.accuracy

            model_SNPE_2.Grassmann_NPE_Object.fit_transform(model_SNPE_2.data, model_SNPE_2.target)
            AR = Analysis_Riemannian(model_SNPE_2.Grassmann_NPE_Object, save_file=False)
            AR.Analysis()
            KNN_2[epoch, i] = AR.knn.accuracy

    KNN_mean_1 = np.nanmean(KNN_1, axis=0)
    KNN_std_1 = np.nanstd(KNN_1, axis=0)

    KNN_mean_2 = np.nanmean(KNN_2, axis=0)
    KNN_std_2 = np.nanstd(KNN_2, axis=0)

    Draw_Line_Chart(
        filename=model_SNPE_1.sec_part + "-" + str(model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-neighbors",
        path="./Figure/" + dn,
        xlabel="The Number of Neighbors $k$",
        ylabel_left="Mean Classification Accuracy",
        column=n_list,
        left=[KNN_mean_1, KNN_mean_2],
        ylim_left=(0, 1.05),
        left_marker=("d", "*"),
        left_color=["#999A9E", "#8A7197"],
        left_label=["GSNPE-I", "GSNPE-II"]).Draw_simple_line()

    EB = Error_Drawing(
        path="./Figure/" + dn,
        xlabel="The Number of Neighbors $k$",
        ylabel="Mean Classification Accuracy")

    EB.filename = model_SNPE_1.sec_part + "-" + str(
        model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-neighbors-banding"
    EB.drawing_banding(
        x_value=np.array(n_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("r", "g"),
        markers=("P", "X"), labels=["GSNPE-I", "GSNPE-II"])

    EB.filename = model_SNPE_1.sec_part + "-" + str(
        model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-neighbors-errorbar"
    EB.drawing_bar_error(
        x_value=np.array(n_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("#E6846D", "#8DCDD5"), labels=["GSNPE-I", "GSNPE-II"])

    EB.filename = model_SNPE_1.sec_part + "-" + str(
        model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-neighbors-line-error"
    EB.drawing_line_error(
        x_value=np.array(n_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("#F9BEBB", "#89C9C8"), labels=["GSNPE-I", "GSNPE-II"])

    EB.filename = model_SNPE_1.sec_part + "-" + str(
        model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-neighbors-barh-error"
    EB.drawing_barh_error(
        x_value=np.array(n_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("#E44A33", "#4DBAD6"), labels=["GSNPE-I", "GSNPE-II"])

    gc.collect()

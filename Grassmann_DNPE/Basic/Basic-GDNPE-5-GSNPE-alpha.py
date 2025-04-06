################################################################################
## 基础实验5：Grassmann Semi-Supervised NPE-I的alpha寻优
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
alpha_list = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
################################################################################
## 运行实验
for dn in config.basic_data:
    KNN_1 = np.zeros((max_epoch, len(alpha_list)))
    KNN_2 = np.zeros((max_epoch, len(alpha_list)))
    for epoch in range(max_epoch):
        config.random_state = np.random.randint(2024)
        for i, alpha in enumerate(alpha_list):
            model_SNPE_1 = factory(
                func_name='GSNPE',
                data_name=dn,
                return_time=config.return_time,
                train_size=config.train_size.get(dn),
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=5)

            model_SNPE_2 = factory(
                func_name='GSNPE',
                data_name=dn,
                return_time=config.return_time,
                train_size=config.train_size.get(dn),
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=5)

            model_SNPE_1.Product_Grassmann_NPE_Object(
                n_components=config.low_dimensions.get(dn),
                n_neighbors=config.n_neighbors,
                p_grassmann=config.grassmann_p.get(dn),
                converged_tol=config.converged_tol,
                max_epoch=config.max_epoch,
                alpha=alpha,
                mode=1,
                is_discriminant=config.is_discriminant.get(model_SNPE_1.func_name),
                is_semi_supervised=config.is_semi_supervised.get(model_SNPE_1.func_name),
                is_self_supervised=config.is_self_supervised.get(model_SNPE_1.func_name))

            model_SNPE_2.Product_Grassmann_NPE_Object(
                n_components=config.low_dimensions.get(dn),
                n_neighbors=config.n_neighbors,
                p_grassmann=config.grassmann_p.get(dn),
                converged_tol=config.converged_tol,
                max_epoch=config.max_epoch,
                alpha=alpha,
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
        filename=model_SNPE_1.sec_part + "-" + str(model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-alpha",
        path="./Figure/" + dn,
        xlabel="Regularization Coefficient " + chr(945),
        ylabel_left="Mean Classification Accuracy",
        column=alpha_list,
        left=[KNN_mean_1, KNN_mean_2],
        ylim_left=(0, 1.05),
        left_marker=("P", "X"),
        left_color=["#8CA3C3", "#D2ADA8"],
        left_label=["GSNPE-I", "GSNPE-II"]).Draw_simple_line()

    EB = Error_Drawing(
        path="./Figure/" + dn,
        xlabel="Regularization Coefficient " + chr(945),
        ylabel="Mean Classification Accuracy")

    EB.filename = model_SNPE_1.sec_part + "-" + str(model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-alpha-banding"
    EB.drawing_banding(
        x_value=np.array(alpha_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("r", "g"),
        markers=("P", "X"), labels=["GSNPE-I", "GSNPE-II"])

    EB.filename = model_SNPE_1.sec_part + "-" + str(model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-alpha-errorbar"
    EB.drawing_bar_error(
        x_value=np.array(alpha_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("#E6846D", "#8DCDD5"), labels=["GSNPE-I", "GSNPE-II"])

    EB.filename = model_SNPE_1.sec_part + "-" + str(
        model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-alpha-line-error"
    EB.drawing_line_error(
        x_value=np.array(alpha_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("#A32A31", "#407BD0"), labels=["GSNPE-I", "GSNPE-II"])

    EB.filename = model_SNPE_1.sec_part + "-" + str(
        model_SNPE_1.sec_num) + '-' + model_SNPE_1.func_name + "-" + dn + "-best-alpha-barh-error"
    EB.drawing_barh_error(
        x_value=np.array(alpha_list),
        mean_value=[KNN_mean_1, KNN_mean_2],
        std_value=[KNN_std_1, KNN_std_2],
        colors=("#E44A33", "#4DBAD6"), labels=["GSNPE-I", "GSNPE-II"])

    gc.collect()

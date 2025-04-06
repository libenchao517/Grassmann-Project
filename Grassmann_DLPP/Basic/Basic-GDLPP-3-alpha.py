################################################################################
## 基础实验3：Grassmann SLPP的alpha探索
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
from GDLPP import GrassmannDLPP_config
import gc
################################################################################
## 定义基本变量
config = GrassmannDLPP_config()
max_epoch = 5
alpha_list = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
mean_knns = []
std_knns = []
################################################################################
## 运行实验
for dn in config.basic_data:
    KNN_ = np.zeros((max_epoch, len(alpha_list)))
    for epoch in range(max_epoch):
        config.random_state = np.random.randint(2024)
        for i, alpha in enumerate(alpha_list):
            model_SLPP = factory(
                func_name='GSLPP',
                data_name=dn,
                return_time=config.return_time,
                train_size=config.train_size.get(dn),
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=3)

            model_SLPP.Product_Grassmann_LPP_Object(
                n_components=config.low_dimensions.get(dn),
                n_neighbors=config.n_neighbors,
                p_grassmann=config.grassmann_p.get(dn),
                alpha=None,
                converged_tol=config.converged_tol,
                max_epoch=config.max_epoch,
                is_discriminant=config.is_discriminant.get(model_SLPP.func_name),
                is_semi_supervised=config.is_semi_supervised.get(model_SLPP.func_name),
                is_self_supervised=config.is_self_supervised.get(model_SLPP.func_name))

            model_SLPP.Grassmann_LPP_Object.alpha = alpha
            model_SLPP.Grassmann_LPP_Object.fit_transform(model_SLPP.data, model_SLPP.target)
            AR = Analysis_Riemannian(model_SLPP.Grassmann_LPP_Object, save_file=False)
            AR.Analysis(classification=True)
            KNN_[epoch, i] = AR.knn.accuracy

    KNN_mean = np.nanmean(KNN_, axis=0)
    KNN_std = np.nanstd(KNN_, axis=0)

    mean_knns.append(KNN_mean)
    std_knns.append(KNN_std)

    gc.collect()

Draw_Line_Chart(
    filename=model_SLPP.sec_part + "-" + str(model_SLPP.sec_num) + "-" + model_SLPP.func_name + "-" + "-alpha",
    xlabel="Regularization Coefficient " + chr(958),
    ylabel_left="Mean Classification Accuracy",
    column=alpha_list,
    left=mean_knns,
    ylim_left=(0, 1.05),
    left_marker=["^", "v", "<", ">"],
    left_color=["#A4C8D9", "#6C96CC", "#EDAE92", "#C92321"],
    left_label=config.basic_data).Draw_simple_line()

EB = Error_Drawing(
    xlabel="Regularization Coefficient " + chr(958),
    ylabel="Mean Classification Accuracy")

EB.filename = model_SLPP.sec_part + "-" + str(
    model_SLPP.sec_num) + '-' + model_SLPP.func_name + "-best-alpha-errorbar"
EB.drawing_bar_error(
    x_value=np.array(alpha_list),
    mean_value=mean_knns,
    std_value=std_knns,
    colors=["#A4C8D9", "#6C96CC", "#EDAE92", "#C92321"],
    labels=config.basic_data)

EB.filename = model_SLPP.sec_part + "-" + str(
    model_SLPP.sec_num) + '-' + model_SLPP.func_name + "-best-alpha-line-error"
EB.drawing_line_error(
    x_value=np.array(alpha_list),
    mean_value=mean_knns,
    std_value=std_knns,
    colors=["#A4C8D9", "#6C96CC", "#EDAE92", "#C92321"],
    labels=config.basic_data)

EB.filename = model_SLPP.sec_part + "-" + str(
        model_SLPP.sec_num) + '-' + model_SLPP.func_name + "-best-alpha-barh-error"
EB.drawing_barh_error(
    x_value=np.array(alpha_list),
    mean_value=mean_knns,
    std_value=std_knns,
    colors=["#A4C8D9", "#6C96CC", "#EDAE92", "#C92321"],
    labels=config.basic_data)

################################################################################
## 基础实验4：Grassmann SLPP的近邻数探索
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
nn_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
mean_knns = []
std_knns = []
################################################################################
## 运行实验
for dn in config.basic_data:
    KNN_ = np.zeros((max_epoch, len(nn_list)))
    for epoch in range(max_epoch):
        config.random_state = np.random.randint(2024)
        for i, nn in enumerate(nn_list):
            model_SLPP = factory(
                func_name='GSLPP',
                data_name=dn,
                return_time=config.return_time,
                train_size=config.train_size.get(dn),
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=4)

            model_SLPP.Product_Grassmann_LPP_Object(
                n_components=config.low_dimensions.get(dn),
                n_neighbors=None,
                p_grassmann=config.grassmann_p.get(dn),
                converged_tol=config.converged_tol,
                max_epoch=config.max_epoch,
                is_discriminant=config.is_discriminant.get(model_SLPP.func_name),
                is_semi_supervised=config.is_semi_supervised.get(model_SLPP.func_name),
                is_self_supervised=config.is_self_supervised.get(model_SLPP.func_name))

            model_SLPP.Grassmann_LPP_Object.n_neighbors = nn
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
    filename=model_SLPP.sec_part + "-" + str(
        model_SLPP.sec_num) + "-" + model_SLPP.func_name + "-neighbors",
    xlabel="Neighbors $k$",
    ylabel_left="Mean Classification Accuracy",
    column=nn_list,
    left=mean_knns,
    ylim_left=(0, 1.05),
    left_marker=("d", "P", "X", "*"),
    left_color=["#DB432C", "#438870", "#838AAF", "#C4B797"],
    left_label=config.basic_data
).Draw_simple_line()

EB = Error_Drawing(
    xlabel="Neighbors $k$",
    ylabel="Mean Classification Accuracy")

EB.filename = model_SLPP.sec_part + "-" + str(
    model_SLPP.sec_num) + '-' + model_SLPP.func_name + "-best-neighbors-errorbar"
EB.drawing_bar_error(
    x_value=np.array(nn_list),
    mean_value=mean_knns,
    std_value=std_knns,
    colors=["#DB432C", "#438870", "#838AAF", "#C4B797"],
    labels=config.basic_data)

EB.filename = model_SLPP.sec_part + "-" + str(
    model_SLPP.sec_num) + '-' + model_SLPP.func_name + "-best-neighbors-line-error"
EB.drawing_line_error(
    x_value=np.array(nn_list),
    mean_value=mean_knns,
    std_value=std_knns,
    colors=["#DB432C", "#438870", "#838AAF", "#C4B797"],
    labels=config.basic_data)

EB.filename = model_SLPP.sec_part + "-" + str(
        model_SLPP.sec_num) + '-' + model_SLPP.func_name + "-best-neighbors-barh-error"
EB.drawing_barh_error(
    x_value=np.array(nn_list),
    mean_value=mean_knns,
    std_value=std_knns,
    colors=["#DB432C", "#438870", "#838AAF", "#C4B797"],
    labels=config.basic_data)

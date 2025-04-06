################################################################################
## 参数实验3：Grassmann流形的子空间阶数分析
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
## 导入模块
import gc
import numpy as np
from Assess import Analysis_Rieman_Cluster
from Factory import factory
from GUMAP import GrassmannUMAP_config
from Draw import Error_Drawing
################################################################################
## 定义基本变量
config = GrassmannUMAP_config()
max_epoch = 5
p_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
mean_ari = []
std_ari = []
mean_ami = []
std_ami = []
mean_hms = []
std_hms = []
################################################################################
## 运行实验
for dn in config.basic_data:
    ARI_ = np.zeros((max_epoch, len(p_list)))
    AMI_ = np.zeros((max_epoch, len(p_list)))
    HMS_ = np.zeros((max_epoch, len(p_list)))
    for epoch in range(max_epoch):
        config.random_state = np.random.randint(2024)
        for i, p in enumerate(p_list):
            model = factory(
                func_name='GUMAP',
                data_name=dn,
                return_time=config.return_time,
                train_size=None,
                random_state=config.random_state,
                sec_part='Basic',
                sec_num=3)
            model.Product_Grassmann_UMAP_Object(
                n_components=config.GUMAP_components,
                n_neighbors=config.GUMAP_neighbors,
                p_grassmann=p)
            model.Grassmann_UMAP_Object.fit_transform(model.data, model.target)
            AR = Analysis_Rieman_Cluster(model.Grassmann_UMAP_Object, cluster=["GSC"])
            AR.Analysis()
            ARI_[epoch, i] = AR.ari.score
            AMI_[epoch, i] = AR.ami.score
            HMS_[epoch, i] = AR.hms.score
    ARI_mean = np.nanmean(ARI_, axis=0)
    ARI_std = np.nanstd(ARI_, axis=0)
    mean_ari.append(ARI_mean)
    std_ari.append(ARI_std)

    AMI_mean = np.nanmean(AMI_, axis=0)
    AMI_std = np.nanstd(AMI_, axis=0)
    mean_ami.append(AMI_mean)
    std_ami.append(AMI_std)

    HMS_mean = np.nanmean(HMS_, axis=0)
    HMS_std = np.nanstd(HMS_, axis=0)
    mean_hms.append(HMS_mean)
    std_hms.append(HMS_std)

    EB = Error_Drawing(
        path="./Figure/" + dn,
        fontsize=18,
        titlefontsize=20,
        xlabel="The Orders of Subspace $p$",
        ylabel="Clustering Performance")

    EB.filename = model.sec_part + "-" + str(model.sec_num) + '-' + model.func_name + "-" + dn + "-best-orders-banding"
    EB.drawing_banding(
        x_value=np.array(p_list),
        mean_value=[ARI_mean, AMI_mean, HMS_mean],
        std_value=[ARI_std, AMI_std,HMS_std],
        colors=("r", "g", "b"),
        markers=("<", ">", "^"), labels=["ARI", "AMI", "HMS"])

    EB.filename = model.sec_part + "-" + str(model.sec_num) + '-' + model.func_name + "-" + dn + "-best-orders-line-error"
    EB.drawing_line_error(
        x_value=np.array(p_list),
        mean_value=[ARI_mean, AMI_mean, HMS_mean],
        std_value=[ARI_std, AMI_std,HMS_std],
        colors=("#B83945", "#EDB31E", "#27B2AF"), labels=["ARI", "AMI", "HMS"])

    gc.collect()

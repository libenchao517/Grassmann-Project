################################################################################
## 实验2：Grassmann UMAP降维结果在Grassmann聚类算法的聚类效果测试
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
from Assess import Analysis_Rieman_Cluster
from Factory import factory
from GUMAP import GrassmannUMAP_config
################################################################################
## 定义基本变量
config = GrassmannUMAP_config()
################################################################################
## 运行实验
for dn in config.GUMAP_data:
    model = factory(
        func_name='GUMAP',
        data_name=dn,
        return_time=config.return_time,
        train_size=0.5,
        random_state=config.random_state,
        sec_part='Experiment',
        sec_num=2)
    model.Product_Grassmann_UMAP_Object(
        n_components=config.GUMAP_components,
        n_neighbors=config.GUMAP_neighbors,
        p_grassmann=config.grassmann_p)
    model.Grassmann_UMAP_Object.fit_transform(model.data, model.target)
    Analysis_Rieman_Cluster(model.Grassmann_UMAP_Object).Analysis()

    gc.collect()

################################################################################
## 实验1：Grassmann Discriminant LPP效果测试
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
import gc
################################################################################
## 定义基本变量
config = GrassmannDLPP_config()
################################################################################
## 运行实验
for dn in config.GDLPP_data:
    model = factory(
        func_name='GDLPP',
        data_name=dn,
        return_time=config.return_time,
        train_size=config.train_size.get(dn),
        random_state=config.random_state,
        sec_part='Experiment',
        sec_num=1)
    model.Product_Grassmann_LPP_Object(
        n_components=config.low_dimensions.get(dn),
        n_neighbors=config.n_neighbors,
        p_grassmann=config.grassmann_p.get(dn),
        converged_tol=config.converged_tol,
        max_epoch=config.max_epoch,
        is_discriminant=config.is_discriminant.get(model.func_name),
        is_semi_supervised=config.is_semi_supervised.get(model.func_name),
        is_self_supervised=config.is_self_supervised.get(model.func_name))
    model.Grassmann_LPP_Object.fit_transform(model.data, model.target)
    Analysis_Riemannian(model.Grassmann_LPP_Object).Analysis()
    gc.collect()

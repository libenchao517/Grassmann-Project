################################################################################
# 本代码用于自动化运行GUMAP项目实验
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入模块
import gc
import time
from Utils import Auto_Run
################################################################################
# 运行项目
AR = Auto_Run(
    Project="GUMAP",
    MRPY=None,
    content="Grassmann_UMAP/Basic",
    is_parallel=True,
    lock=True
)
AR.Run()

for i in range(10):
    AR = Auto_Run(
        Project="GUMAP",
        MRPY="Make_Results_GUMAP.py",
        content="Grassmann_UMAP/Experiment",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    gc.collect()

AR = Auto_Run(
    Project="GUMAP",
    MRPY=None,
    content="Results/GUMAP-Project",
    run_file="Total_GUMAP.py",
    is_parallel=False,
    lock=True
)
AR.Run()

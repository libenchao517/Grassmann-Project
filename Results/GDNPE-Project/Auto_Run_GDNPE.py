################################################################################
# 本代码用于自动化运行GDNPE项目实验
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
    Project="GDNPE",
    MRPY="Make_Pre_Results_GDNPE.py",
    content="Grassmann_DNPE/Basic",
    is_parallel=False,
    lock=True
)
AR.Run()

for i in range(10):
    AR = Auto_Run(
        Project="GDNPE",
        MRPY=None,
        content="Grassmann_DNPE/Comparatation",
        is_parallel=False,
        lock=True
    )
    AR.Run()
    gc.collect()

    AR = Auto_Run(
        Project="GDNPE",
        MRPY="Make_Results_GDNPE.py",
        content="Grassmann_DNPE/Experiment",
        is_parallel=False,
        lock=True
    )
    AR.Run()
    gc.collect()

    time.sleep(300)

AR = Auto_Run(
    Project="GDNPE",
    MRPY=None,
    content="Results/GDNPE-Project",
    run_file="Total_GDNPE.py",
    is_parallel=False,
    lock=True
)
AR.Run()


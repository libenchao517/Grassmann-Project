################################################################################
# 本代码用于自动化运行GDLPP项目实验
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
    Project="GDLPP",
    MRPY="Make_Pre_Results_GDLPP.py",
    content="Grassmann_DLPP/Basic",
    is_parallel=False,
    lock=True
)
AR.Run()

gc.collect()

for i in range(10):
    AR = Auto_Run(
        Project="GDLPP",
        MRPY=None,
        content="Grassmann_DLPP/Comparatation",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    gc.collect()

    AR = Auto_Run(
        Project="GDLPP",
        MRPY="Make_Results_GDLPP.py",
        content="Grassmann_DLPP/Experiment",
        is_parallel=False,
        lock=True
    )
    AR.Run()

    gc.collect()

    time.sleep(300)


AR = Auto_Run(
    Project="GDLPP",
    MRPY=None,
    content="Results/GDLPP-Project",
    run_file="Total_GDLPP.py",
    is_parallel=False,
    lock=True
)
AR.Run()

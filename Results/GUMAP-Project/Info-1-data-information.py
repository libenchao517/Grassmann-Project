import numpy as np
import pandas as pd
from DATA import Load_Data
from GUMAP import GrassmannUMAP_config
from GDNPE import GrassmannDNPE_config
from GDLPP import GrassmannDLPP_config
config = GrassmannUMAP_config()
info = pd.DataFrame(
    columns=["samples", "classes", "Gr(p, D)", "Gr(p, d)"],
    index=config.GUMAP_data)
for dn in config.GUMAP_data:
    data, target = Load_Data(dn + "-" + str(config.grassmann_p)).Loading()
    info.loc[dn, "samples"] = data.shape[0]
    info.loc[dn, "classes"] = len(np.unique(target))
    info.loc[dn, "Gr(p, D)"] = f"Gr({data.shape[2]}, {data.shape[1]})"
    info.loc[dn, "Gr(p, d)"] = f"Gr({data.shape[2]}, {config.GUMAP_components})"
info.to_excel("Info-1-data-information.xlsx")

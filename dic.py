import numpy as np
D_name_set = ['Samsung_NV15_0','Samsung_NV15_1','Samsung_NV15_2','Canon_PowerShotA640_0','Canon_Ixus70_2','Nikon_D200_1','Casio_EX_Z150_0','Casio_EX_Z150_1','Casio_EX_Z150_4','Sony_DSC_H50_0_46956']
B_name_set = ['Samsung','Canon','Nikon','Casio','Sony']
B_lable = [13,12,10,0,9]

M_name_set = ['Samsung_NV15','Canon_PowerShotA640','Canon_Ixus70','Nikon_D200','Casio_EX_Z150','Sony_DSC_H50']
M_lable = [[0,0,0,0,0    ,0,0,0,0,0,   0,0,0,1         ,0,0,0,0,1          ,0,0,0,0,0],
[0,0,0,0,0    ,0,0,0,0,0,   0,0,1,0         ,0,0,0,0,1          ,0,0,0,0,0],
[0,0,0,0,0    ,0,0,0,0,0,   0,0,1,0         ,0,0,0,1,0          ,0,0,0,0,0],
[0,0,0,0,0    ,0,0,0,0,0,   1,0,0,0         ,0,0,0,0,1          ,0,0,0,0,0],
[1,0,0,0,0    ,0,0,0,0,0,   0,0,0,0         ,0,0,0,0,1          ,0,0,0,0,0],
[0,0,0,0,0    ,0,0,0,0,1,   0,0,0,0         ,0,0,0,0,1          ,0,0,0,0,0]]
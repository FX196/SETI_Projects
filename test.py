from blimpy import Waterfall
path_to_fil = "spliced_blc0001020304050607_guppi_58100_80372_OUMUAMUA_OFF_0016.gpuspec.0002.fil"
fb = Waterfall(path_to_fil)
print(fb.info())
print(fb.data)
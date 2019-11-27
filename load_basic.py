import os
'''
BASIC SETTINGS
'''

NET_NAME = "test_linear_net"
DATA_FOLDER = "./等温度"
EXPER_FOLDER = "./experData"
DATA_SIZE = len([x for x in os.listdir(DATA_FOLDER) if x.endswith("mod")])
EXPER_SIZE = len([x for x in os.listdir(EXPER_FOLDER) if x.endswith("mod")])

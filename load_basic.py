import os
'''
BASIC SETTINGS
'''



'''
CHANGE THESE BELOW !!!
'''
#  NEW_NET_NAME  = "ANN"
#  NEW_NET_NAME  = "testSave_STD"
#  NEW_NET_NAME = "testSave_standard"
#  NEW_NET_NAME = "testSave_standard_dropout"
NEW_NET_NAME = "ch_standard_conv1d"
#  MATTER_NAME  = "oh"
MATTER_NAME  = "ch"
'''
DO NOT TOUCH THE OTHERS
'''

NET_NAME  = "./model/{}".format(NEW_NET_NAME)
DATA_FOLDER = f"./{MATTER_NAME}"
EXPER_FOLDER = "./experData"
DATA_SIZE = len([x for x in os.listdir(DATA_FOLDER) if x.endswith("mod")])
EXPER_SIZE = len([x for x in os.listdir(EXPER_FOLDER) if x.endswith("mod")])

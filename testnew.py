import re
PATTERN = re.compile("vtemp=(\d+) rtemp=(\d+)")

import os 
for name in os.listdir("ch"):
    v = PATTERN.findall(name)[0][0]
    filename = f"ch vtemp={v.zfill(4)} rtemp={v.zfill(4)}.mod"
    os.rename(f"./ch/{name}", f"./ch/{filename}")

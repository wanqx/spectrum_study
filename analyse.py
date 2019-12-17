import time
import win32gui, win32api, win32con
from keyboardMap import VK_CODE

# hld = win32gui.FindWindow(None, u"OH (A) population distributions")

# left, top, right, bottom = win32gui.GetWindowRect(hld)
# print(left, top, right, bottom)
class GetGUI:
    def __init__(self, name):
        self.WAIT_TIME = 0.002
        self.WAIT_TIME_ACTION = 0.001
        self.hld = win32gui.FindWindow(None, name)
        self.handle = None

    def get_child_windows(self, parent):
        '''
        获得parent的所有子窗口句柄
         返回子窗口句柄列表
         '''
        if not parent:
            return
        hwndChildList = []
        win32gui.EnumChildWindows(parent, lambda hwnd, param: param.append(hwnd),  hwndChildList)
        return hwndChildList

    def test(self):
        for item in get_child_windows(hld):
            title = win32gui.GetWindowText(item)
            if "population distributions" in title:
                print(title)
                computeID = item

    def keyboardInput(self, word, slptime=None):
        '''
        if input a string, make sleep time input else self.WAIT_TIME
        '''
        WAIT_TIME = self.WAIT_TIME if slptime is None else slptime
        if len(word) > 1: print("keyboard input: {0}".format(word))
        win32api.keybd_event(VK_CODE[word], 0, 0, 0)
        time.sleep(WAIT_TIME)
        win32api.keybd_event(VK_CODE[word], 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(WAIT_TIME)

    def keyboardInputString(self, s):
        if not isinstance(s, str): s = str(s)
        print("keyboard input: {0}".format(s))
        time.sleep(self.WAIT_TIME)
        for ab in s:
            self.keyboardInput(ab, self.WAIT_TIME_ACTION)
        time.sleep(self.WAIT_TIME)

    def inputString(self, s):
        if not isinstance(s, str): s = str(s)
        win32api.SendMessage(self.handle, win32con.WM_SETTEXT, 0, s)

    def leftClick(self):
        print("excuting left click")
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(self.WAIT_TIME)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


    def getChildWindow(self, chld, Num, Name=None):
        tmpID = win32gui.FindWindowEx(chld, 0, None, None)
        cnt = 0
        def inner():
            nonlocal cnt, tmpID
            if cnt >= Num: return tmpID
            cnt += 1
            tmpID = win32gui.FindWindowEx(chld, tmpID, None, None)
            return inner()
        return inner()

    def locateWindow(self, *tree):
        '''
        tree = [2,1,2,2,3]
        means that 1 of hld, 0 of child, 1 of child...
        '''
        if len(tree)>=2: tree = tree
        if len(tree)==1: tree = tree[0]
        _tree = [x-1 for x in tree]
        firstNode = _tree[0]
        tmpID = self.getChildWindow(self.hld, firstNode)
        if len(_tree)==1: return tmpID
        for item in _tree[1:]:
            tmpID = eval(f"self.getChildWindow(tmpID, {item})")
        self.handle = tmpID
        print("select handle: {0}".format(hex(tmpID)))
        return tmpID

    def moveCursor(self, *args):
        if len(args)>=2: dx, dy = args
        if len(args)==1: dx, dy = args[0]
        x, y, *_ = win32gui.GetWindowRect(self.handle)
        print("set cursor to {0}, {1}".format(x, y))
        win32api.SetCursorPos((x+dx,y+dy))
        time.sleep(self.WAIT_TIME)

    def clickButton(self, *args):
        time.sleep(self.WAIT_TIME)
        self.moveCursor(*args)
        self.leftClick()
        time.sleep(self.WAIT_TIME)

    def clickButtonDouble(self, *args):
        time.sleep(self.WAIT_TIME)
        self.moveCursor(*args)
        self.leftClick()
        time.sleep(self.WAIT_TIME_ACTION)
        self.leftClick()
        time.sleep(self.WAIT_TIME)



test = GetGUI("LIFBASE  2. 1. 1")

def transferLifbase(GUI, vtemp, rtemp):
    name = "          oh vtemp={0} rtemp={1}".format(vtemp, rtemp)
    test.locateWindow(1, 1) # select Vibration distributions
    test.clickButtonDouble(370, 50) # select Vibration distributions
    test.keyboardInputString(vtemp)
    time.sleep(0.1)
    test.clickButtonDouble(370, 80) # select Rotation distributions
    test.keyboardInputString(rtemp)
    time.sleep(0.1)
    test.locateWindow(2,1,2,2,3) # press button
    test.clickButton(415, 10)
    time.sleep(0.1)

    test.locateWindow(2,1,2,1,1) # save plot
    test.clickButton(50, 10)
    time.sleep(0.1)
    test.keyboardInputString(name)
    time.sleep(0.15)
    test.keyboardInput("enter")
    time.sleep(1.75)

    #  test.clickButton(70, 10)
    #  test.clickButton(70, 150)
    #  test.keyboardInputString(name)
    #  test.keyboardInput("enter")
    #  time.sleep(0.8)

from timer import timer
@timer
def main():
    #  for i in range(1000, 5000, 10):
        #  for j in range(1000, 5000, 10):
            #  transferLifbase(test, vtemp=i, rtemp=j)
    for i in range(1583, 6011, 1):
        transferLifbase(test, vtemp=i, rtemp=i)

main()

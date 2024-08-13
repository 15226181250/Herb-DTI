## 导航栏 ##
import wx
import win32api
import webbrowser
from PIL import Image
import linecache
import os
import numpy as np
import sys
def _CreateMenuBar(self):
    '''创建菜单栏'''
    self.mb = wx.MenuBar()
    # 设置菜单
    m = wx.Menu()
    # m.Append(self.id_help, u"帮助主题")
    m.Append(self.id_fontSize, u"日志字体大小")
    m.Append(self.id_fontStyle, u"日志字体样式")
    m.Append(self.id_saveInput, u"是否保存记录")
    m.Append(self.id_windowT, u"设置窗口透明度")
    self.mb.Append(m, u"|  系统设置  |")
    # self.Bind(wx.EVT_MENU, self.OnHelp,id=self.id_help)
    # self.Bind(wx.EVT_MENU, self.OnFontSize,id=self.id_fontSize)
    # self.Bind(wx.EVT_MENU, self.OnFontStyle,id=self.id_fontStyle)
    # self.Bind(wx.EVT_MENU, self.OnSaveInput,id=self.id_saveInput)
    # self.Bind(wx.EVT_MENU, self.OnWindowT,id=self.id_windowT)
    # 关于菜单
    m = wx.Menu()
    # m.Append(self.id_help, u"帮助主题")
    m.Append(self.id_about, u"研究室简介")
    m.Append(self.id_people, u"开发者信息")
    m.Append(self.id_vx, u"联系我们")
    self.mb.Append(m, u"|  关于我们  |")
    # self.Bind(wx.EVT_MENU, self.OnHelp,id=self.id_help)
    # self.Bind(wx.EVT_MENU, self.OnAbout,id=self.id_about)
    # self.Bind(wx.EVT_MENU, self.OnPeople,id=self.id_people)
    # self.Bind(wx.EVT_MENU, self.OnVx,id=self.id_vx)
    # 帮助菜单
    m = wx.Menu()
    m.Append(self.id_help, u"帮助文档")
    m.Append(self.id_note, u"简易便签")
    m.Append(self.id_justChooseDir, u"打开最近选择的文件夹")
    m.Append(self.id_justSaveDir, u"打开最近保存的文件夹")
    #m.Append(self.id_save, u"保存文件")
    m.AppendSeparator()
    m.Append(self.id_quit, u"退出系统")
    self.mb.Append(m, u"|  帮助工具  |")
    # self.Bind(wx.EVT_MENU, self.OnHelp, id=self.id_help)
    # self.Bind(wx.EVT_MENU, self.OpenNote, id=self.id_note)
    # self.Bind(wx.EVT_MENU, self.JustChooseDir, id=self.id_justChooseDir)
    # self.Bind(wx.EVT_MENU, self.JustSaveDir, id=self.id_justSaveDir)
    # #self.Bind(wx.EVT_MENU, self.OnSave, id=self.id_save)
    # self.Bind(wx.EVT_MENU, self.OnQuit, id=self.id_quit)
    self.SetMenuBar(self.mb)
def _CreateStatusBar(self):
    '''创建状态栏'''
    self.sb = self.CreateStatusBar()
    self.sb.SetFieldsCount(1)
    self.sb.SetStatusWidths([-1])
    self.sb.SetStatusStyles([wx.SB_RAISED])
    self.sb.SetStatusText(self.bottom_line, 0)
    # self.sb = self.CreateStatusBar()
    # self.sb.SetFieldsCount(1)
    # # self.sb.SetStatusWidths([-1, -3, -1])
    # self.sb.SetStatusStyles([wx.SB_RAISED])
def OnQuit(self, evt):
    '''退出系统'''
    self.Destroy()
def OnHelp(self, evt):
    '''帮助'''
    path01 = os.path.dirname(sys.argv[0])
    os.startfile(path01 + r'/不可删除/help_file.pdf')
def OnAbout(self, evt):
    '''关于'''
    webbrowser.open("https://mp.weixin.qq.com/s/cXMDRDMcDXZuH6Abs9C86g")
def OnVx(self, evt):
    '''联系'''
    path01 = os.path.dirname(sys.argv[0])
    # os.startfile(path01 + r'/不可删除/Vx.pdf')
    img=Image.open(path01 + r'/不可删除/Vx.png')
    img.show()
def OnPeople(self, evt):
    dlg = wx.MessageDialog(None, u"总体设计：项荣武\n开发人员：李天成  申镇华\n美工设计：申镇华  李定远\n开发团队：沈阳药科大学数据与信息科学研究室", u"开发者介绍",  wx.OK | wx.ICON_QUESTION)
    if dlg.ShowModal() == wx.OK:
        self.Close(True)
    dlg.Destroy()
def pngSize(pic,width,hight):
    p = wx.Image(pic, wx.BITMAP_TYPE_PNG).ConvertToBitmap()  # 载入图片
    img = p.ConvertToImage()
    bgm = img.Scale(width,hight)
    return wx.Bitmap(bgm)
def EvtText(self, event):
    self.tempNum01 = event.GetString()
def EvtComboBox(self, event):
    self.tempNum01 = event.GetString()
def OnFontSize(self, event):
    print('aaaaaddfdf')
    dlg = wx.SingleChoiceDialog(None, u"请选择日志字体大小:", u"设置日志字体大小",
                                ['8','9', '10', '11', '12', '13', '14','15','16'])
    dlg.SetSize((240, 300))
    dlg.Center()
    if dlg.ShowModal() == wx.ID_OK:
        message01 = dlg.GetStringSelection() #获取选择的内容
        f=open('不可删除/information.txt','r+',encoding='utf-8')
        flist=f.readlines()
        flist[0]= message01 + '\n'
        f=open('不可删除/information.txt','w+',encoding='utf-8')
        f.writelines(flist)
        f.flush()
        f.close()
        self.logger.AppendText('[WARNING] ' + "请您退出，重新启动系统生效!!!" + '\n\n')
        dlg = wx.MessageDialog(None, "请您退出，重新启动系统生效!!!", u"系统提示",  wx.OK | wx.ICON_INFORMATION) #  提示  wx.ICON_INFORMATION  非提示 wx.ICON_QUESTION
        if dlg.ShowModal() == wx.OK:
            self.Close(True)
        dlg.Destroy()
    dlg.Destroy()
def OnFontStyle(self, event):
    dlg = wx.SingleChoiceDialog(None, u"请选择日志字体样式:", u"设置日志字体样式",
                                ['仿宋', '宋体', '黑体', '幼圆', '楷体','新宋体', '微软雅黑','华文仿宋','方正姚体'])
    dlg.SetSize((240, 300))
    dlg.Center()
    if dlg.ShowModal() == wx.ID_OK:
        message02 = dlg.GetStringSelection() #获取选择的内容
        f=open('不可删除/information.txt','r+',encoding='utf-8')
        flist2=f.readlines()
        flist2[1]= message02 + '\n'
        f=open('不可删除/information.txt','w+',encoding='utf-8')
        f.writelines(flist2)
        f.flush()
        f.close()
        self.logger.AppendText('[WARNING] ' + "请您退出，重新启动系统生效!!!" + '\n\n')
        dlg = wx.MessageDialog(None, "请您退出，重新启动系统生效!!!", u"系统提示",  wx.OK | wx.ICON_INFORMATION) #  提示  wx.ICON_INFORMATION  非提示 wx.ICON_QUESTION
        if dlg.ShowModal() == wx.OK:
            self.Close(True)
        dlg.Destroy()
    dlg.Destroy()
def OnSaveInput(self, event):
    dlg = wx.SingleChoiceDialog(None, u"是否保存输入记录:", u"设置保存输入记录",
                                ['是', '否'])
    dlg.SetSize((240, 300))
    dlg.Center()
    if dlg.ShowModal() == wx.ID_OK:
        message02 = dlg.GetStringSelection() #获取选择的内容
        f=open('不可删除/information.txt','r+',encoding='utf-8')
        flist2=f.readlines()
        flist2[3]= message02 + '\n'
        f=open('不可删除/information.txt','w+',encoding='utf-8')
        f.writelines(flist2)
        f.flush()
        f.close()
        self.logger.AppendText('[WARNING] ' + "请您退出，重新启动系统生效!!!" + '\n\n')
        dlg = wx.MessageDialog(None, "请您退出，重新启动系统生效!!!", u"系统提示",  wx.OK | wx.ICON_INFORMATION) #  提示  wx.ICON_INFORMATION  非提示 wx.ICON_QUESTION
        if dlg.ShowModal() == wx.OK:
            self.Close(True)
        dlg.Destroy()
    dlg.Destroy()
def OnWindowT(self, event):
    dlg = wx.SingleChoiceDialog(None, u"请选择窗口透明度:", u"设置窗口透明",
                                ['不透明', '240', '230', '220', '210', '200'])
    dlg.SetSize((240, 300))
    dlg.Center()
    if dlg.ShowModal() == wx.ID_OK:
        message02 = dlg.GetStringSelection() #获取选择的内容
        f=open('不可删除/information.txt','r+',encoding='utf-8')
        flist2=f.readlines()
        flist2[14]= message02 + '\n'
        f=open('不可删除/information.txt','w+',encoding='utf-8')
        f.writelines(flist2)
        f.flush()
        f.close()
        self.logger.AppendText('[WARNING] ' + "请您退出，重新启动系统生效!!!" + '\n\n')
        dlg = wx.MessageDialog(None, "请您退出，重新启动系统生效!!!", u"系统提示",  wx.OK | wx.ICON_INFORMATION) #  提示  wx.ICON_INFORMATION  非提示 wx.ICON_QUESTION
        if dlg.ShowModal() == wx.OK:
            self.Close(True)
        dlg.Destroy()
    dlg.Destroy()
def JustChooseDir(self, event):
    linecache.clearcache()
    inputD08 = linecache.getline('不可删除/information.txt',16)
    start_directory01 = inputD08.strip()
    if start_directory01 == '' :
        self.logger.AppendText('[WARNING] ' + "刚刚没有选择文件夹打开!" + '\n\n')
        dlg = wx.MessageDialog(None, u'刚刚没有选择文件夹打开!', u'操作提示', wx.OK | wx.ICON_QUESTION)
        dlg.SetFont(wx.Font(9, family = wx.DEFAULT, style = wx.NORMAL, weight = wx.BOLD, faceName = 'Consolas'))
        dlg.SetBackgroundColour(wx.Colour(224, 574, 324))
        dlg.Show()
        if(dlg.ShowModal() == wx.ID_OK):
            arr = np.array('f')
            print(arr[4])
    else :
        os.startfile(start_directory01)
def JustSaveDir(self, event):
    linecache.clearcache()
    inputD09 = linecache.getline('不可删除/information.txt',17)
    start_directory02 =  inputD09.strip()
    if start_directory02 == '' :
        self.logger.AppendText('[WARNING] ' + "刚刚没有选择文件夹打开!" + '\n\n')
        dlg = wx.MessageDialog(None, u'刚刚没有选择文件夹打开!', u'操作提示', wx.OK | wx.ICON_QUESTION)
        dlg.SetFont(wx.Font(9, family = wx.DEFAULT, style = wx.NORMAL, weight = wx.BOLD, faceName = 'Consolas'))
        dlg.SetBackgroundColour(wx.Colour(224, 574, 324))
        dlg.Show()
        if(dlg.ShowModal() == wx.ID_OK):
            arr = np.array('f')
            print(arr[4])
    else :
        os.startfile(start_directory02)
def OpenNote(self, event):
    MainWindow(None, "简易便签")
class MainWindow(wx.Frame):
    """
    记事本
    """
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title)
        self.control = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.dibu001 = self.CreateStatusBar()  # 创建位于窗口的底部的状态栏
        self.dibu001.SetFieldsCount(1)
        self.dibu001.SetStatusStyles([wx.SB_RAISED])
        self.SetSize((440, 660))
        font1 = wx.Font(11, family = wx.DEFAULT, style = wx.NORMAL, weight = wx.NORMAL, faceName = '微软雅黑')
        self.control.SetFont(font1)
        self.dibu001.SetStatusText(u'     Copyright ©2022-2025 沈阳药科大学数据与信息科学研究室版权所有', 0)
        # self.icon = wx.Icon('one.ico', wx.BITMAP_TYPE_ICO)
        # self.SetIcon(self.icon)
        self.Center()
        self.Bind(wx.EVT_CLOSE, self.OnClose002)
        f = open('不可删除/bianqian.txt', 'r', encoding='utf-8')
        self.control.SetValue(f.read())
        f.flush()
        f.close()
        # dlg.Destroy()
        self.Show(True)
def OnClose002(self, evt):
    '''关闭窗口事件函数'''
    dlg = wx.MessageDialog(None, u'是否保存对便签的修改内容？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
    dlg.SetFont(wx.Font(9, family = wx.DEFAULT, style = wx.NORMAL, weight = wx.BOLD, faceName = 'Consolas'))
    dlg.SetBackgroundColour(wx.Colour(224, 574, 324))
    dlg.Show()
    if(dlg.ShowModal() == wx.ID_YES):
        file = open('不可删除/bianqian.txt', 'w', encoding='utf-8')
        file.write(self.control.GetValue())
        file.flush()
        file.close()
        self.Destroy()
    else:
        self.Destroy()





















import wx
import os
import menuBarTool
from tkinter import filedialog
import tool.Herb2vec.mainFunction
import tool.SMILESynergy.mainFunction as Smain
import main.modelMain as modelMain
import argparse
import tool.DeepGOPlus.predict as DGPpredict
class MyFrame(wx.Frame):
    id_open = wx.NewId()
    id_save = wx.NewId()
    id_quit = wx.NewId()
    id_help = wx.NewId()
    id_about = wx.NewId()
    id_aa = wx.NewId()
    id_vx = wx.NewId()
    id_people = wx.NewId()
    id_fontSize = wx.NewId()
    id_fontStyle = wx.NewId()
    id_saveInput = wx.NewId()
    id_windowT = wx.NewId()
    id_justChooseDir = wx.NewId()
    id_justSaveDir = wx.NewId()
    id_note = wx.NewId()
    def __init__(self, parent, title):
        self.dirname=''
        wx.Frame.__init__(self, parent, title=title)
        self.SetBackgroundColour(wx.Colour(250, 250, 250))
        #self.SetBackgroundColour('white')
        # image_width = to_bmp_image.GetWidth()
        # image_height = to_bmp_image.GetHeight()
        # 控制主窗口大小
        self.SetSize((1000, 618))
        self.SetMaxSize((1000, 618))
        self.SetMinSize((1000, 618))
        # TODO 右侧
        self.sizer3 = wx.BoxSizer(wx.VERTICAL)
        # 右侧选择框
        self.panel = wx.Panel(parent=self)
        image_file = 'static/picture/backgroundPhoto.jpg'
        self.to_bmp_image = wx.Image(image_file, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.staticBackground = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.staticText = wx.StaticText(parent=self.staticBackground,
                                        label='    欢迎使用药物靶点相互作用预测工具\nHerb-DTI，该工具开发的目的是辅助非计\n算机专业人员使用深度学习来预测中草药\n化合物和疾病靶点相互作用关系，具体使\n用方法请仔细阅读帮助工具里面的使用说\n明。该工具严禁商用，谢谢合作。',
                                        pos=(40,100))
        font1 = wx.Font(16, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='宋体')
        self.staticText.SetFont(font1)
        self.staticText.SetBackgroundColour(wx.Colour(230, 226, 215))
        self.sizer3.Add(self.panel, 22, wx.EXPAND)
        self.button1Background = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.button1Background.Hide()
        self.button2Background = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.button2Background.Hide()
        self.button3Background = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.button3Background.Hide()
        self.button4Background = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.button4Background.Hide()
        self.button5Background = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.button5Background.Hide()
        self.button6Background = wx.StaticBitmap(self.panel, -1, self.to_bmp_image, (0,0),(736,366))
        self.button6Background.Hide()
        # 右侧控制台
        self.logger = wx.TextCtrl(self, style=wx.TE_MULTILINE,size=(300,100))
        self.sizer3.Add(self.logger, 10, wx.EXPAND)
        # TODO 左侧
        self.sizer4 = wx.BoxSizer(wx.VERTICAL)
        # 左侧功能栏
        self.sizer2 = self._CreateFunctionBar()
        self.sizer4.Add(self.sizer2, 4, wx.EXPAND)
        # 左侧选择栏下方
        image_file = './static/picture/backgroundPhoto.png'
        to_bmp_image = wx.Image(image_file, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.bitmap = wx.StaticBitmap(self, -1, to_bmp_image, (0,0),(10,165))
        self.sizer4.Add(self.bitmap, 1, wx.EXPAND)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.sizer4, 1, wx.EXPAND)
        self.sizer.Add(self.sizer3, 3, wx.EXPAND)
        self.CreateStatusBar()
        # TODO 创建顶部菜单栏
        menuBarTool._CreateMenuBar(self)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        self.Show()
    # TODO 左侧功能栏
    def _CreateFunctionBar(self):
        font2 = wx.Font(13, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.button1 = wx.Button(self, -1, "数据标准化")
        self.button1.SetFont(font2)
        self.button1.SetBackgroundColour(wx.Colour(122, 80, 68))
        sizer2.Add(self.button1, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.button1Click, self.button1)
        self.button2 = wx.Button(self, -1, "表型特征转向量")
        self.button2.SetFont(font2)
        self.button2.SetBackgroundColour(wx.Colour(160, 119, 107))
        sizer2.Add(self.button2, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.button2Click, self.button2)
        self.button3 = wx.Button(self, -1, "Smiles式转向量")
        self.button3.SetFont(font2)
        self.button3.SetBackgroundColour(wx.Colour(122, 80, 68))
        sizer2.Add(self.button3, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.button3Click, self.button3)
        self.button4 = wx.Button(self, -1, "蛋白质序列转向量")
        self.button4.SetFont(font2)
        self.button4.SetBackgroundColour(wx.Colour(160, 119, 107))
        sizer2.Add(self.button4, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.button4Click, self.button4)
        self.button5 = wx.Button(self, -1, "训练DTI模型")
        self.button5.SetFont(font2)
        self.button5.SetBackgroundColour(wx.Colour(122, 80, 68))
        sizer2.Add(self.button5, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.button5Click, self.button5)
        self.button6 = wx.Button(self, -1, "DTI预测")
        self.button6.SetFont(font2)
        self.button6.SetBackgroundColour(wx.Colour(160, 119, 107))
        sizer2.Add(self.button6, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.button6Click, self.button6)
        return sizer2
    # TODO 数据标准化
    def button1Click(self,event):
        # self.staticText.Destroy() # 删除指定控件
        self.staticBackground.Hide() # 隐藏指定控件
        self.button2Background.Hide()
        self.button3Background.Hide()
        self.button4Background.Hide()
        self.button5Background.Hide()
        self.button6Background.Hide()
        self.button1Background.Show()
        self.staticText = wx.StaticText(parent=self.button1Background,
                                        label='      请确用来训练模型的药物数据和靶点数据符合以下\n格式，点击以下相应按钮查看。',
                                        pos=(20,50))
        font1 = wx.Font(12, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        self.staticText.SetFont(font1)
        self.staticText.SetBackgroundColour(wx.Colour(230, 226, 215))
        font3 = wx.Font(10, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        self.lblname111 = wx.StaticText(self.button1Background, label="查看药物示范数据:", pos=(70,150))
        # self.editname111 = wx.TextCtrl(self.button1Background, value="", pos=(50, 280), size=(140,-1))
        self.lblname112 = wx.StaticText(self.button1Background, label="查看靶点示范数据:", pos=(280,150))
        # self.editname112 = wx.TextCtrl(self.button1Background, value="", pos=(250, 280), size=(140,-1))
        self.button111 = wx.Button(self.button1Background, -1, "药物数据",pos=(83, 200))
        self.button111.SetFont(font3)
        self.button111.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseResultFile11, self.button111)
        self.button112 = wx.Button(self.button1Background, -1, "靶点数据",pos=(294, 200))
        self.button112.SetFont(font3)
        self.button112.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseResultFile12, self.button112)
    def ChooseResultFile11(self,event):
        os.startfile('D:\\ProjectWarehouse\\PythonProject\\Herb-DTI\\data\\entityData\\drug')
    def ChooseResultFile12(self,event):
        os.startfile('D:\\ProjectWarehouse\\PythonProject\\Herb-DTI\\data\\entityData\\target')
    # TODO 表型特征转向量
    def button2Click(self,event):
        self.staticBackground.Hide() # 隐藏指定控件
        self.button1Background.Hide()
        self.button3Background.Hide()
        self.button4Background.Hide()
        self.button5Background.Hide()
        self.button6Background.Hide()
        self.button442 =wx.Button(self.button2Background, label="开始训练", pos=(180, 325))
        # self.editname = wx.TextCtrl(self.button2Background, value="6666666", pos=(50, 80), size=(140,-1))
        self.button2Background.Show()
        self.axiom_file_path = '../data/mateData/Herb2vec/axiom/axiomsorig.lst'
        self.association_file= '../data/phenotypeData/drug/drug_association_file'
        self.outfile2 = '../model/phenotypeFeature/sideEffectEmbeddingModel'
        self.embedding_size = 200
        self.entityList2 = '../../data/phenotypeData/drug/drug_list'
        self.num_workers = 12
        font3 = wx.Font(10, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        wx.StaticText(self.button2Background, label='关联文件内容为实体和实体表型特征间的映射', pos=(50,80))
        self.button21 = wx.Button(self.button2Background, -1, "选择关联文件",pos=(300, 80))
        self.button21.SetFont(font3)
        self.button21.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseAssociationFile, self.button21)
        wx.StaticText(self.button2Background, label='公理文件内容为实体表型特征间的各种关系', pos=(50,110))
        self.button22 = wx.Button(self.button2Background, -1, "选择公理文件",pos=(300, 110))
        self.button22.SetFont(font3)
        self.button22.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseAxiomFile, self.button22)
        wx.StaticText(self.button2Background, label='输出文件需要空的文件', pos=(50,140))
        self.button23 = wx.Button(self.button2Background, -1, "选择输出文件",pos=(300, 140))
        self.button23.SetFont(font3)
        self.button23.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseOutFile2, self.button23)
        wx.StaticText(self.button2Background, label='实体文件内容为需要转换的实体(药物\蛋白质)', pos=(50,170))
        self.button24 = wx.Button(self.button2Background, -1, "选择实体文件",pos=(300, 170))
        self.button24.SetFont(font3)
        self.button24.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseEntityList, self.button24)
        self.sampleList21 = ['100', '200 (默认)', '300', '400', '500']
        self.lblhear21 = wx.StaticText(self.button2Background, label="选择生成向量大小：", pos=(50, 205))
        self.embedding_size = wx.ComboBox(self.button2Background, pos=(170, 200), size=(80, -1), choices=self.sampleList21, style=wx.CB_DROPDOWN)
        self.sampleList22 = ['1', '8 (默认)', '16', '24', '32', '48']
        self.lblhear22 = wx.StaticText(self.button2Background, label="选择工作线程数目：", pos=(50, 245))
        self.num_workers = wx.ComboBox(self.button2Background, pos=(170, 240), size=(80, -1), choices=self.sampleList22, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_BUTTON, self.OnClick2,self.button442)
    def ChooseAssociationFile(self,event):
        file_pre = '../../process/Herb2vecCache/'
        files_names = filedialog.askopenfilenames(title='选择关联文件',initialdir=file_pre)
        self.association_file = files_names
    def ChooseAxiomFile(self,event):
        file_pre = '../../process/Herb2vecCache/'
        files_names = filedialog.askopenfilenames(title='选择要校验的文件',initialdir=file_pre)
        self.axiom_file_path = files_names
    def ChooseOutFile2(self,event):
        file_pre = '../../process/Herb2vecCache/'
        files_names = filedialog.askopenfilenames(title='选择要校验的文件',initialdir=file_pre)
        self.outfile2 = files_names
    def ChooseEntityList(self,event):
        file_pre = '../../process/Herb2vecCache/'
        files_names = filedialog.askopenfilenames(title='选择要校验的文件',initialdir=file_pre)
        self.entityList2 = files_names
    # TODO Smiles式转向量
    def button3Click(self,event):
        # self.staticText.Destroy() # 删除指定控件
        self.staticBackground.Hide() # 隐藏指定控件
        self.button2Background.Hide()
        self.button1Background.Hide()
        self.button4Background.Hide()
        self.button5Background.Hide()
        self.button6Background.Hide()
        self.button443 =wx.Button(self.button3Background, label="开始训练", pos=(180, 325))
        self.button3Background.Show()
        font3 = wx.Font(10, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        self.num_workers3 = 12
        wx.StaticText(self.button3Background, label='选择需要转换的化合物Smiles式列表：', pos=(50,100))
        self.button31 = wx.Button(self.button3Background, -1, "选择输入文件",pos=(300, 100))
        self.button31.SetFont(font3)
        self.button31.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseAssociationFile3, self.button31)
        wx.StaticText(self.button3Background, label='输出文件需要空的文件', pos=(50,140))
        self.button33 = wx.Button(self.button3Background, -1, "选择输出文件",pos=(300, 140))
        self.button33.SetFont(font3)
        self.button33.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseOutFile3, self.button33)
        self.sampleList31 = ['trfm  (默认)', 'rnn']
        self.lblhear31 = wx.StaticText(self.button3Background, label="生成化合物的嵌入（向量表示）的模型：", pos=(50, 180))
        self.modeling_methods3 = wx.ComboBox(self.button3Background, pos=(305, 175), size=(80, -1), choices=self.sampleList31, style=wx.CB_DROPDOWN)
        self.sampleList32 = ['1', '8 (默认)', '16', '24', '32', '48']
        self.lblhear32 = wx.StaticText(self.button3Background, label="选择工作线程数目：", pos=(50, 220))
        self.num_workers3 = wx.ComboBox(self.button3Background, pos=(305, 215), size=(80, -1), choices=self.sampleList32, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_BUTTON, self.OnClick3,self.button443)
    def ChooseAssociationFile3(self,event):
        file_pre3 = '../../process/Herb2vecCache/'
        files_names3 = filedialog.askopenfilenames(title='选择关联文件',initialdir=file_pre3)
        self.association_file3 = files_names3
    def ChooseOutFile3(self,event):
        file_pre = '../../process/Herb2vecCache/'
        files_names = filedialog.askopenfilenames(title='选择要校验的文件',initialdir=file_pre)
        self.outfile3 = files_names
    def button4Click(self,event):
        # self.staticText.Destroy() # 删除指定控件
        self.staticBackground.Hide() # 隐藏指定控件
        self.button2Background.Hide()
        self.button3Background.Hide()
        self.button1Background.Hide()
        self.button5Background.Hide()
        self.button6Background.Hide()
        self.button444 =wx.Button(self.button4Background, label="开始训练", pos=(180, 325))
        self.button4Background.Show()
        font3 = wx.Font(10, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        self.num_workers4 = 12
        wx.StaticText(self.button4Background, label='选择需要转换的蛋白质序列列表：', pos=(50,90))
        self.button41 = wx.Button(self.button4Background, -1, "选择输入文件",pos=(300, 90))
        self.button41.SetFont(font3)
        self.button41.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseAssociationFile4, self.button41)
        wx.StaticText(self.button4Background, label='输出文件需要空的文件', pos=(50,120))
        self.button43 = wx.Button(self.button4Background, -1, "选择输出文件",pos=(300, 120))
        self.button43.SetFont(font3)
        self.button43.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseOutFile4, self.button43)
        self.sampleList41 = ['100', '500', '1000 (默认)', '1500', '2000']
        self.lblhear41 = wx.StaticText(self.button4Background, label="选择一次读取的序列数：", pos=(50, 165))
        self.chunk_size4 = wx.ComboBox(self.button4Background, pos=(305, 160), size=(80, -1), choices=self.sampleList41, style=wx.CB_DROPDOWN)
        self.sampleList42 = ['16', '32', '64', '128 (默认)', '256']
        self.lblhear42 = wx.StaticText(self.button4Background, label="选择预测模型的批量大小：", pos=(50, 195))
        self.batch_size4 = wx.ComboBox(self.button4Background, pos=(305, 190), size=(80, -1), choices=self.sampleList42, style=wx.CB_DROPDOWN)
        self.sampleList43 = ['0.3', '0.4', '0.5 (默认)', '0.6', '0.7']
        self.lblhear43 = wx.StaticText(self.button4Background, label="选择Alpha权重参数：", pos=(50, 225))
        self.alpha4 = wx.ComboBox(self.button4Background, pos=(305, 220), size=(80, -1), choices=self.sampleList43, style=wx.CB_DROPDOWN)
        self.sampleList44 = ['1', '8 (默认)', '16', '24', '32', '48']
        self.lblhear44 = wx.StaticText(self.button4Background, label="选择工作线程数目：", pos=(50, 255))
        self.num_workers4 = wx.ComboBox(self.button4Background, pos=(305, 250), size=(80, -1), choices=self.sampleList44, style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_BUTTON, self.OnClick4,self.button444)
    def ChooseAssociationFile4(self,event):
        file_pre3 = '../../process/Herb2vecCache/'
        files_names3 = filedialog.askopenfilenames(title='选择关联文件',initialdir=file_pre3)
        self.association_file4 = files_names3
    def ChooseOutFile4(self,event):
        file_pre = '../../process/Herb2vecCache/'
        files_names = filedialog.askopenfilenames(title='选择要校验的文件',initialdir=file_pre)
        self.outfile4 = files_names
        self.Bind(wx.EVT_BUTTON, self.OnClick4,self.button44)
    def button5Click(self,event):
        # self.staticText.Destroy() # 删除指定控件
        self.staticBackground.Hide() # 隐藏指定控件
        self.button2Background.Hide()
        self.button3Background.Hide()
        self.button4Background.Hide()
        self.button1Background.Hide()
        self.button6Background.Hide()
        self.button445 =wx.Button(self.button5Background, label="开始训练", pos=(180, 335))
        self.lblname5 = wx.StaticText(self.button5Background, label="请确保嵌入模型已训练完成！！！", pos=(150,5))
        self.button5Background.Show()
        font3 = wx.Font(10, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        self.sampleList51 = ['GCNConv (默认)', 'GENConv', 'GATConv', 'APPNP']
        self.lblhear51 = wx.StaticText(self.button5Background, label="选择训练模型的深度学习方法：", pos=(50, 35))
        self.deep_learning_methods5 = wx.ComboBox(self.button5Background, pos=(305, 30), size=(80, -1), choices=self.sampleList51, style=wx.CB_DROPDOWN)
        self.sampleList52 = ['20', '50', '100', '200 (默认)', '500']
        self.lblhear52 = wx.StaticText(self.button5Background, label="选择建模过程中训练多少批次：", pos=(50, 65))
        self.num_epochs5 = wx.ComboBox(self.button5Background, pos=(305, 60), size=(80, -1), choices=self.sampleList52, style=wx.CB_DROPDOWN)
        self.sampleList53 = ['16', '32 (默认)', '64', '128', '256']
        self.lblhear53 = wx.StaticText(self.button5Background, label="选择每个批次数据集的大小：", pos=(50, 95))
        self.batch_size5 = wx.ComboBox(self.button5Background, pos=(305, 90), size=(80, -1), choices=self.sampleList53, style=wx.CB_DROPDOWN)
        self.sampleList54 = ['2', '4', '5 (默认)', '10', '20']
        self.lblhear54 = wx.StaticText(self.button5Background, label="选择交叉验证的折叠次数：", pos=(50, 125))
        self.numFolds = wx.ComboBox(self.button5Background, pos=(305, 120), size=(80, -1), choices=self.sampleList54, style=wx.CB_DROPDOWN)
        self.sampleList55 = ['0.1', '0.01', '0.001', '0.0001 (默认)', '0.00001']
        self.lblhear55 = wx.StaticText(self.button5Background, label="选择构建模型的学习率：", pos=(50, 155))
        self.learning_rate = wx.ComboBox(self.button5Background, pos=(305, 150), size=(80, -1), choices=self.sampleList55, style=wx.CB_DROPDOWN)

        self.sampleList56 = ['1', '2', '3 (默认)', '4', '5']
        self.lblhear56 = wx.StaticText(self.button5Background, label="选择建模过程中更新模型的频数：", pos=(50, 185))
        self.update_frequency= wx.ComboBox(self.button5Background, pos=(305, 180), size=(80, -1), choices=self.sampleList56, style=wx.CB_DROPDOWN)

        self.sampleList57 = ['500', '600', '700 (默认)', '800', '900']
        self.lblhear57 = wx.StaticText(self.button5Background, label="选择PPI网络的可信得分阈值：", pos=(50, 215))
        self.trusted_score_threshold5 = wx.ComboBox(self.button5Background, pos=(305, 210), size=(80, -1), choices=self.sampleList57, style=wx.CB_DROPDOWN)

        self.sampleList58 = ['true (默认)', 'false']
        self.lblhear58 = wx.StaticText(self.button5Background, label="选择建模过程是否包含分子结构特征：", pos=(50, 245))
        self.molecular_structural_features5 = wx.ComboBox(self.button5Background, pos=(305, 240), size=(80, -1), choices=self.sampleList58, style=wx.CB_DROPDOWN)

        self.sampleList59 = ['trfm (默认)', 'rnn']
        self.lblhear59 = wx.StaticText(self.button5Background, label="选择构建化合物分子嵌入模型的方法：", pos=(50, 275))
        self.modeling_methods5 = wx.ComboBox(self.button5Background, pos=(305, 270), size=(80, -1), choices=self.sampleList59, style=wx.CB_DROPDOWN)

        self.sampleList511 = ['1', '8 (默认)', '16', '24', '32', '48']
        self.lblhear511 = wx.StaticText(self.button5Background, label="选择工作线程数目：", pos=(50, 305))
        self.num_workers5 = wx.ComboBox(self.button5Background, pos=(305, 300), size=(80, -1), choices=self.sampleList511, style=wx.CB_DROPDOWN)

        self.button55 = wx.Button(self.button5Background, -1, "查看模型",pos=(450, 335))
        self.button55.SetFont(font3)
        self.button55.SetBackgroundColour(wx.Colour(122, 80, 68))
        self.Bind(wx.EVT_BUTTON, self.ChooseResultFile5, self.button55)


        self.Bind(wx.EVT_BUTTON, self.OnClick5,self.button445)

    def ChooseResultFile5(self,event):
        os.startfile('D:\\ProjectWarehouse\\PythonProject\\Herb-DTI\\model\\resultDTIModel')

        # self.Bind(wx.EVT_BUTTON, self.OnClick4,self.button44)




    def button6Click(self,event):
        # self.staticText.Destroy() # 删除指定控件
        self.staticBackground.Hide() # 隐藏指定控件
        self.button2Background.Hide()
        self.button3Background.Hide()
        self.button4Background.Hide()
        self.button5Background.Hide()
        self.button1Background.Hide()
        self.button446 =wx.Button(self.button6Background, label="开始预测", pos=(180, 325))
        self.button6Background.Show()

        self.staticText = wx.StaticText(parent=self.button6Background,
                                        label='      使用该模块之前请确保已完成DTI模型的构建，请\n按照如下格式要求输入数据，进行药物靶点相互作用\n预测。\n               药物格式：CIDm00000323\n               靶点格式：9606.ENSP00000306512\n      结果为0-1，越接近于1则两者之间具有相互作用的\n可能性越大。',
                                        pos=(20,50))
        font1 = wx.Font(12, family=wx.DEFAULT, style=wx.NORMAL, weight=wx.BOLD, faceName='微软雅黑')
        self.staticText.SetFont(font1)
        self.staticText.SetBackgroundColour(wx.Colour(230, 226, 215))

        self.lblname661 = wx.StaticText(self.button6Background, label="输入目标药物:", pos=(70,250))
        self.editname661 = wx.TextCtrl(self.button6Background, value="", pos=(50, 280), size=(140,-1))
        self.lblname662 = wx.StaticText(self.button6Background, label="输入目标靶点:", pos=(280,250))
        self.editname662 = wx.TextCtrl(self.button6Background, value="", pos=(250, 280), size=(140,-1))

        self.Bind(wx.EVT_BUTTON, self.OnClick6,self.button446)


    def OnClick(self,event):
        self.logger.AppendText(" Click on object with Id %d\n" %event.GetId())


    # def OnClick2(self,association_file,axiom_file,entity_list,outfile,embedding_size,file_pre,num_workers):
    def OnClick2(self,event):
        # print(self.association_file[0])
        # print(self.axiom_file_path[0])
        # print(self.outfile2[0])
        # print(self.entityList2[0])
        # print(self.embedding_size.GetStringSelection())
        # print(self.num_workers.GetStringSelection())
        self.logger.AppendText("开始为实体的表型特征生成向量...\n")
        self.G2 = tool.Herb2vec.mainFunction.generate_graph(str(self.association_file[0]), str(self.axiom_file_path[0]))
        self.logger.AppendText("已生成关系网络："+str(self.G2))
        self.logger.AppendText("随机游走中，请耐心等待...\n")
        print('33333333333333333------------------------------------------333333333333333')
        tool.Herb2vec.mainFunction.getNodeVector(self.G2, inputFline=str(self.entityList2[0]),
                                                 outputFile=str(self.outfile2[0]),
                                                 embedding_size=int(self.embedding_size.GetStringSelection()),
                                                 file_pre='../process/Herb2vecCache/',
                                                 num_workers=int(self.num_workers.GetStringSelection()))
        self.logger.AppendText("已为实体的表型特征生成向量，请检查指定输出文件。\n")


    def OnClick3(self,event):
        print(111)
        self.logger.AppendText("开始计算药物分子嵌入\n这可能需要一段时间，请耐心等待...\n")
        drug_list=[]
        with open(str(self.association_file3[0]), "r") as f:
            for line in f.readlines():
                drug_list.append(line.strip())
        Smain.generateDrugVector(drug_list, mode=str(self.modeling_methods3.GetStringSelection()))
        self.logger.AppendText("成功计算出药物分子嵌入。正在中止脚本的其余部分\n")


    def OnClick4(self,event):
        self.logger.AppendText("加载参数中...\n")
        # @ck.option('--in-file', '-if', help='Input FASTA file')
        # @ck.option('--out-file', '-of', default='./results.tsv', help='Output result file')
        # @ck.option('--go-file', '-gf', default='../../data/mateData/DeepGOPlus/data/go.obo', help='Gene Ontology')
        # @ck.option('--model-file', '-mf', default='../../data/mateData/DeepGOPlus/data/model.h5', help='Tensorflow model file')
        # @ck.option('--terms-file', '-tf', default='../../data/mateData/DeepGOPlus/data/terms.pkl', help='List of predicted terms')
        # @ck.option('--annotations-file', '-tf', default='../../data/mateData/DeepGOPlus/data/train_data.pkl', help='Experimental annotations')
        # @ck.option('--chunk-size', '-cs', default=1000, help='Number of sequences to read at a time')
        # @ck.option('--diamond-file', '-df', default='data/test_diamond.res', help='Diamond Mapping file')
        # @ck.option('--threshold', '-t', default=0.0, help='Prediction threshold')
        # @ck.option('--batch-size', '-bs', default=128, help='Batch size for prediction model')
        # @ck.option('--alpha', '-a', default=0.5, help='Alpha weight parameter')
        self.logger.AppendText("开始计算蛋白质序列嵌入\n这可能需要一段时间，请耐心等待...\n")
        while(True):
            pass
        # DGPpredict.main()
        self.logger.AppendText("成功计算出蛋白质序列嵌入。正在中止脚本的其余部分\n")


    def OnClick5(self,event):
        self.logger.AppendText("加载参数中...\n")

        parser = argparse.ArgumentParser()

        # 给parser实例添加属性  parser.add_argument()
        parser.add_argument("--numProteins", type=int, default=-1)
        parser.add_argument("--arch", type=str, default='GENConv')
        parser.add_argument("--node_features", type=str, default='MolPred')

        parser.add_argument("--num_epochs", type=int, default=200)

        # parser.add_argument("--batchSize", type=int, default=13)
        parser.add_argument("--batchSize", type=int, default=20)

        parser.add_argument("--numFolds", type=int, default=5)
        parser.add_argument("--lr", type=float, default=0.0001)

        parser.add_argument("--fold", type=int, default=3)

        # parser.add_argument("--drug_mode", type=str, default='trfm')
        # parser.add_argument("--include_mol_features", action='store_true')

        parser.add_argument("--herbTest", type=str, default='false')
        parser.add_argument("--mode", type=str, default='')
        parser.add_argument("--PPIMinScore", type=int, default=700)
        parser.add_argument("--includeMolecularFeatures", type=str, default='true')
        parser.add_argument("--drugMode", type=str, default='trfm')

        # 通过config = parser.parse_args()把刚才的属性从parser给config，后面直接通过config使用
        config = parser.parse_args()

        self.logger.AppendText("开始训练药物靶点相互作用预测模型\n这可能需要一段时间，请耐心等待...\n")

        import main.modelMain as modelMain555
        modelMain555.quickenedMissingTargetPredictor(config)

    def OnClick6(self,event):
        self.logger.AppendText('您选择的目标药物是：'+str(self.editname661.GetValue())+'\n')
        self.logger.AppendText('您选择的目标靶点是：'+str(self.editname662.GetValue())+'\n')
        self.logger.AppendText("开始预测目标化合物和目标靶点之间的相互作用...\n")

        import torch
        file_path = "D:\\ProjectWarehouse\\PythonProject\\Herb-DTI\\model\resultDTIModel\\PPI_network_model_with_mol_features_fold_3.model"
        model = torch.load(file_path)
        signal666 = ''
        for key, value in model.items():
            if key[0] == str(self.editname661.GetValue()) and key[1] == str(self.editname662.GetValue()):
                signal666 = key[2]
            break
        if signal666 != '':
            self.logger.AppendText('目标化合物和目标靶点之间的相互作用可能性为：'+str(signal666)+'\n')
        else:
            self.logger.AppendText('目标化合物或目标靶点未识别，请更新预测模型'+'\n')



















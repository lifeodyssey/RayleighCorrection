import glob
import math
import os
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker
import numpy as np
import rasterio as rio
from PIL import Image
from Py6S import *
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from osgeo import gdal, gdalconst, gdal_array
from scipy import interpolate, ndimage

from frame import logger
from sensor import H8
from sensor import GK2A


# 色彩转换，从16进制到R,G,B
# value：‘#7C00FF’
# return：[124,0,255]
def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


# 获取tif影像data数组
# input_file:tif文件路径
# bands：[0,1,2]获取tif文件波段列表，不传默认所有波段
# ignoreValue：获取数组是忽略的无效值，自动变为Nan，不传默认0为无效值
# return：当bands为一个波段时返回二维数组，多个波段时返回二维数组列表
def get_data(input_file, bands=None, ignoreValue=0):
    dataset = gdal.Open(input_file)
    if dataset == None:
        print('无法读取tif文件')
        sys.exit(1)

    if bands == None:
        if dataset.RasterCount == 1:
            banddata = np.array(dataset.GetRasterBand(1).ReadAsArray())
            # 将无效值转化为Nan
            banddata = np.where(banddata != ignoreValue, banddata, np.nan)
            return banddata
        else:
            bandsdata = []
            for i in range(1, dataset.RasterCount + 1):
                banddata = np.array(dataset.GetRasterBand(i).ReadAsArray())
                # 将无效值转化为Nan
                banddata = np.where(banddata != ignoreValue, banddata, np.nan)
                bandsdata.append(banddata)
            return bandsdata
    else:
        if len(bands) == 1:
            banddata = np.array(dataset.GetRasterBand(bands[0]).ReadAsArray())
            # 将无效值转化为Nan
            banddata = np.where(banddata != ignoreValue, banddata, np.nan)
            return banddata
        else:
            bandsdata = []
            for band in bands:
                banddata = np.array(dataset.GetRasterBand(band).ReadAsArray())
                # 将无效值转化为Nan
                banddata = np.where(banddata != 0, banddata, np.nan)
                bandsdata.append(banddata)
            return bandsdata


# 根据数组写tif文件
# filename:需要写出的tif文件路径
# im_proj：写出tif投影
# im_geotrans：写出tif地理坐标
# im_data：写出tif数组，支持二维数组，三维数组
def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetProjection(im_proj)
    dataset.SetGeoTransform(im_geotrans)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])


# 合并多个tif为一个tif
# input_files：输入需要合并的多个tif文件路径
# outputFile：输出合并后的tif文件路径
def meta_spectral(input_files, output_file, ignoreValue=0):
    # 读文件中的数据
    data = []
    for input_file in input_files:
        dataset = gdal.Open(input_file)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        banddata = dataset.ReadAsArray(0, 0, im_width, im_height)
        banddata = np.where(banddata != ignoreValue, banddata, np.nan)
        data.append(banddata)

    write_data = np.array(data, dtype=data[0].dtype)
    write_img(output_file, im_proj, im_geotrans, write_data)


# 将多个大小、位置不一致的tif合成一张
# inputFiles:输入的tif文件列表
# saveFile：输出文件保存路径
def layer_stack(inputFiles, saveFile):
    logger.info(f'layer_stack- executing before')
    outvrt = saveFile.replace('.tif', '.vrt')
    #
    outds = gdal.BuildVRT(outvrt, inputFiles, xRes=0.01, yRes=0.01, separate=True)
    # InputImage = gdal.Open(outvrt, 0)  # open the VRT in read-only mode
    gdal.Translate(saveFile, outds, format='GTiff')
    # creationOptions=['COMPRESS:DEFLATE', 'TILED:YES']
    # os.remove(outvrt)
    # del InputImage
    del outvrt
    logger.info(f'layer_stack- executing end')


# 从tif文件获取经纬度信息
# inputFile：输入要获取经纬度的tif文件
# return：longitude, latitude数组
def Tiff_to_LonLat(inputFile):
    dataset = gdal.Open(inputFile)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    longitude = []
    latitude = []
    # data_1 = dataset.ReadAsArray(0, 0, im_width_1, im_height_1).astype(np.float64)
    for i in range(im_height):
        for j in range(im_width):
            px = im_geotrans[0] + j * im_geotrans[1] + i * im_geotrans[2]
            py = im_geotrans[3] + j * im_geotrans[4] + i * im_geotrans[5]
            longitude.append(px)
            latitude.append(py)
    longitude = np.array(longitude)
    longitude = longitude.reshape(im_height, im_width)
    latitude = np.array(latitude)
    latitude = latitude.reshape(im_height, im_width)
    return longitude, latitude


# 对数据进行辐射定标 return = data*gain+add
# gains：乘常数
# adds：加常数
# data：数组，若输入data为二维数组list，gains，adds也应为列表
# retuen：数组，定标后的数组列表
def radiometricCalibration_bandsdata(gains, adds, data):
    if not isinstance(data, list):
        data = data * gains + adds
        print('辐射定标完成')
        return data
    else:
        new_data = []
        for n_data, gain, add in zip(data, gains, adds):
            n_data = n_data * gain + add
            new_data.append(n_data)
        print('辐射定标完成')
        return new_data


# 对Landsat8数组进行大气校正  方法：Image -Based Atmospheric Correction Revisited and Improved Photogrammetric Engineering and
# year：数组成像时间年
# month：数组成像时间月
# day：数组成像时间日
# data：待校正数组
# retuen：数组，校正后的数组列表
def atmospheric_correction(year, month, day, data):
    # 计算JD
    JD = day - 32075 + 1461 * (year + 4800 + (month - 14) / 12) / 4 + \
         367 * (month - 2 - (month - 14) / 12 * 12) / 12 - 3 * (
                 (year + 4900 + (month - 14) / 12) / 100) / 4
    # 设置ESUNI值
    ESUNI = 196.9
    # 计算日地距离Dist
    Dist = 1 - 0.01674 * math.cos((0.9856 * (JD - 4) * math.pi / 180))
    # 计算太阳天顶角
    cos = math.cos(math.radians(90 - 39.5))
    inter = (math.pi * Dist * Dist) / (ESUNI * cos * cos)
    # 大气校正参数设置
    Lmini = -6.2
    Lmax = 293.7
    Qcal = 1
    Qmax = 255
    LIMIN = Lmini + (Qcal * (Lmax - Lmini) / Qmax)
    LI = (0.01 * ESUNI * cos * cos) / (math.pi * Dist * Dist)
    Lhazel = LIMIN - LI

    if not isinstance(data, list):
        data_ac = inter * (data - Lhazel)
        print('大气校正完成')
        return data_ac
    else:
        data_ac = []
        for n_data in data:
            n_data_ac = inter * (n_data - Lhazel)
            data_ac.append(n_data_ac)
        print('大气校正完成')
        return data_ac


# 对数组进行瑞利校正
# bandsdata：待校正三维数组
# bands：与三维数组对应的波段list：['640']
# solarZenith:与数组对应的太阳天顶角
# solarAzimuth:与数组对应的太阳方位角
# satelliteZenith: 与数组对应的卫星天顶角
# satelliteAzimuth: 与数组对应的卫星方位角
# month: 成像月份
# day: 成像天
# sensor：数据源，目前支持‘H8’，‘GK2A’
# return: 数组，校正后的Rrc数组
def rayleigh_correction(bandsdata, bands, solarZenith, solarAzimuth, satelliteZenith, satelliteAzimuth, month, day,
                        sensor):
    R = np.ones_like(bandsdata)
    s = SixS()
    # 传感器类型 自定义,
    s.geometry = Geometry.User()
    s.geometry.month = month
    s.geometry.day = day

    # ImageCenterLat = centerLat

    # 大气模式类型
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)

    # 气溶胶类型大陆
    s.aero_profile = AtmosProfile.PredefinedType(AeroProfile.NoAerosols)

    # 下垫面类型
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)

    # 研究区海拔、卫星传感器轨道高度
    s.altitudes = Altitudes()
    s.altitudes.set_target_custom_altitude(0)
    s.altitudes.set_sensor_satellite_level()

    for i in range(solarZenith.shape[0]):
        for j in range(17, solarZenith.shape[1]):

            if np.isnan(solarZenith[i, j]):
                continue
            else:
                s.geometry.solar_z = solarZenith[i, j]
                s.geometry.solar_a = solarAzimuth[i, j]
                s.geometry.view_z = satelliteZenith[i, j]
                s.geometry.view_a = satelliteAzimuth[i, j]
                for k in range(len(bands)):
                    band = bands[k]
                    if sensor == 'H8':
                        s.wavelength = Wavelength((H8.H8.himawari_predefined_wavelength[band])[1],
                                                  (H8.H8.himawari_predefined_wavelength[band])[2],
                                                  (H8.H8.himawari_predefined_wavelength[band])[3])
                    elif sensor == 'GK2A':
                        if band == '640':
                            s.wavelength = Wavelength((H8.H8.himawari_predefined_wavelength[band])[1],
                                                      (H8.H8.himawari_predefined_wavelength[band])[2],
                                                      (H8.H8.himawari_predefined_wavelength[band])[3])
                        else:
                            s.wavelength = Wavelength((GK2A.GK2A.GK2A_predefined_wavelength[band])[1],
                                                      (GK2A.GK2A.GK2A_predefined_wavelength[band])[2],
                                                      (GK2A.GK2A.GK2A_predefined_wavelength[band])[3])
                    s.run()
                    Rr = s.outputs.apparent_reflectance
                    R[k, i, j] = Rr

    Rrc = bandsdata - R
    return Rrc


# tif影像重投影
# inputFile：待投影的tif影像路径
# projection：需要转换的重投影类型：'EPSG:3857'
# outputFile：投影后的tif影像路径
# ignoreValue：输出文件时，忽略的无效值，默认为0
def reprojection(inputFile, projection, outputFile, ignoreValue=0):
    dataset = gdal.Open(inputFile)
    warp = gdal.Warp(outputFile, dataset, dstNodata=ignoreValue, dstSRS=projection)
    warp = None
    print('TIF文件转换投影完成')


# 根据输入矢量对tif影像进行裁剪
# inputTifFile：输入待裁剪的Tif文件
# inputShpFile：用于裁剪的shp矢量
# outputTifFile：裁剪后的tif存放路径
# ignoreValue：输出文件时，忽略的无效值，默认为0
def subset_by_shape(inputTifFile, inputShpFile, outputTifFile, ignoreValue=0):
    dataset = gdal.Open(inputTifFile)
    warp = gdal.Warp(outputTifFile, dataset,
                     format='GTiff',
                     cutlineDSName=inputShpFile,
                     cropToCutline=True,
                     dstNodata=ignoreValue)
    warp = None
    print('目标区域裁剪完成')


# 实现对tif影像的重采样，同时输入扩大倍数widthScale，heighScale，或者xRes，yRes
# inputFile：输入原始的Tif文件
# saveFile：要保存的Tif文件路径
# widthScale：图像宽度需要扩大的倍数，例如:2,图像宽度变为原来的2倍，和heighScale同时使用
# heighScale：图像高度度需要扩大的倍数，例如:2,图像搞度变为原来的2倍，和heighScale同时使用
# xRes；图像x轴每个像元的分辨率
# yRes：图像y轴每个像元的分辨率
# resampleAlg：图像的重采样方式，默认为Bilinear，0：NearestNeighbour，
#                                           1：Bilinear:，
#                                           2：Cubic
def resize(inputFile, saveFile, widthScale=None, heighScale=None, xRes=None, yRes=None, resampleAlg=None):
    input_dataset = gdal.Open(inputFile)
    input_GeoTransform = input_dataset.GetGeoTransform()
    if xRes == None & yRes == None & (widthScale is not None) & (heighScale is not None):
        xRes = input_GeoTransform[1] / widthScale
        yRes = input_GeoTransform[-1] / heighScale
    elif widthScale == None & heighScale == None & (xRes is not None) & (yRes is not None):
        xRes = xRes
        yRes = yRes

    if resampleAlg is None:
        resampleAlg = gdalconst.GRA_Bilinear
    elif resampleAlg == 0:
        resampleAlg = gdalconst.GRA_NearestNeighbour
    elif resampleAlg == 1:
        resampleAlg = gdalconst.GRA_Bilinear
    elif resampleAlg == 2:
        resampleAlg = gdalconst.GRA_Cubic

    options = gdal.WarpOptions(options=['tr'], xRes=xRes, yRes=yRes, resampleAlg=resampleAlg)
    gdal.Warp(saveFile, input_dataset, options=options)


# 实现对png图片的重采样
# inputFile：输入待采用的png图片
# saveFile：输入采样后的png图片
# width：图片重采样后的宽度
# heigh：图片重采样后的高度
# resampleAlg：默认为Bilinear，0：NearestNeighbour，
# #                          1：Bilinear:，
# #                          2：Cubic
def png_resize(inputFile, saveFile, width, heigh, resampleAlg=None):
    if resampleAlg is None:
        resampleAlg = Image.BILINEAR
    elif resampleAlg == 0:
        resampleAlg = Image.NEAREST
    elif resampleAlg == 1:
        resampleAlg = Image.BILINEAR
    elif resampleAlg == 2:
        resampleAlg = Image.BICUBIC

    image = Image.open(inputFile)
    image = image.resize((width, heigh), resampleAlg)
    image.save(saveFile)


# 给定tif文件作专题图
# inputFile ： 输入要作图的tif影像
# shpFile ： 输入要作图的shp矢量数据
# saveFile ： png保存路径（包含文件名）
# titleName ： 标题名
# companyName ：附加信息：如公司、数据来源等，放在右下角
# clevs ： 渲染分割数组：例如np.linspace(-10, 15, 6)，最小值为-10，最大值为15，分6类
# locator ： 绘制格网的每格经纬度范围，长度为2的数组
# colorTable ： 颜色数组，格式为16进制
# cmap ： 由颜色创建出来的Colormap，可传默认的颜色表。当传colorTable时不生效
# unit ：单位
# ignoreValue ：图像上的无效值，读取tif时使用
# extend ： 强制范围缩放，默认四个字符串值：'neither'、'min'、'max'、'both';含义：
#          'neither':大于最大值的会显示空白，小于最小值的会显示空白，
#          'min':大于最大值的会显示空白，小于最小值的会显示最小值的颜色，
#          'max':大于最大值的会显示最大值的颜色，小于最小值的会显示空白，
#          'both':大于最大值的会显示最大值的颜色，小于最小值的会显示最小值的颜色，
def draw_tiffile(inputFile, shpFile, saveFile, titleName, companyName, clevs, locator, colorTable=None, cmap=None,
                 unit=None, ignoreValue=0, extend='neither'):
    DATA = get_data(inputFile, ignoreValue=ignoreValue)
    # DATA[DATA>30000] = np.nan
    # DATA[DATA <-10] = np.nan
    ds = rio.open(inputFile)
    MINLON = ds.bounds[0]
    MINLAT = ds.bounds[1]
    MAXLON = ds.bounds[2]
    MAXLAT = ds.bounds[3]
    Extent = [MINLON - 0.1, MAXLON + 0.1, MINLAT - 0.1, MAXLAT + 0.1]
    logger.info(f'draw_tiffile- ds.bounds is {ds.bounds} Extent is {Extent}')
    # 初始化画板，定义坐标范围和投影
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111, projection=proj)

    cityReader = shpreader.Reader(shpFile)
    cityLayer = cfeature.ShapelyFeature(cityReader.geometries(), proj, facecolor='none', linewidths=1,
                                        edgecolors='#A9A9A9')
    fig.subplots_adjust(left=0.07, bottom=0, right=1, top=1, hspace=0, wspace=0)
    ax1.add_feature(cityLayer)

    lons = np.linspace(MINLON, MAXLON, DATA.shape[1])
    lats = np.linspace(MINLAT, MAXLAT, DATA.shape[0])[::-1]

    if colorTable:
        cmap = LinearSegmentedColormap.from_list('chaos', colorTable)

    map_car = ax1.contourf(lons, lats, DATA, clevs, transform=proj, extend=extend,
                           cmap=cmap)

    logger.info(f'draw_tiffile- myfont1 set is Font/msyh.ttc ')
    myfont1 = FontProperties(fname='./Font/msyh.ttc')  # 设置字体
    logger.info(f'draw_tiffile- myfont1 is {myfont1}')

    cbar = plt.colorbar(map_car, fraction=0.08)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    logger.info(f'draw_tiffile- cbar is {cbar}')

    logger.info(f'draw_tiffile- ax1 is {ax1},  Extent is  {Extent}, proj is {proj}')
    # 绘制经纬度格网
    ax1.set_extent(extents=Extent, crs=proj)

    logger.info(f'draw_tiffile- proj is  {proj}')
    gl = ax1.gridlines(alpha=0.5, linestyle='--', draw_labels=True,
                       dms=True, x_inline=False, y_inline=False, )
    logger.info(f'draw_tiffile- gl is  {gl}')
    gl.top_labels = 1
    gl.left_labels = 1
    gl.xlocator = mticker.MultipleLocator(locator[0])
    gl.ylocator = mticker.MultipleLocator(locator[1])
    gl.xformatter = LongitudeFormatter(number_format='.1f', degree_symbol='')
    gl.yformatter = LatitudeFormatter(number_format='.1f', degree_symbol='')
    gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'family': 'Times New Roman', }
    gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'family': 'Times New Roman', }

    TitleColor = "#000000"
    # TimeTitleColor = "#000000"

    plt.rcParams['axes.unicode_minus'] = False

    logger.info(f'draw_tiffile- plt title {titleName}, {TitleColor}, {myfont1}')

    plt.title(titleName, color=TitleColor,
              fontsize=14,
              fontweight=1000, horizontalalignment='center', fontproperties=myfont1)
    CompanyColor = "#000000"

    CompanySize = 12
    # plt.text(0.9, 0, '单位:DU', fontsize=13, color='#000000',fontproperties=myfont1)
    # if (extend =='both') | (extend == 'min'):
    #     unit_pos = [Extent[1] + float(locator[0]) * 0.55, Extent[2] - float(locator[1]) * 0.85]
    # else:
    unit_pos = [Extent[1] + float(locator[0]) * 0.65, Extent[2] - float(locator[1]) * 0.25]

    logger.info(f'draw_tiffile- unit_pos  {unit_pos}')

    if unit != None:
        plt.text(unit_pos[0], unit_pos[1], unit, fontsize=CompanySize, color=CompanyColor,
                 fontweight="bold",
                 horizontalalignment='right', fontproperties=myfont1)
    plt.text(Extent[1] - 0.07, Extent[2] + 0.03, companyName, fontsize=CompanySize, color=CompanyColor,
             fontweight="bold",
             horizontalalignment='right', fontproperties=myfont1)
    # plt.show()
    logger.info(f'draw_tiffile- plt.savefig saveFile  {saveFile}')
    plt.savefig(saveFile, dpi=200, bbox_inches='tight')


# 使用指定的阈值将大于阈值的部分提取出来,输出为红色png，小于阈值部分透明
# 返回像元个数
# inputFile：输入tif文件
# saveFile：保存的PNG文件路径
# threshold：用于提取的阈值
# ignoreValue：读取tif文件时，忽略的无效值，默认为0
# return：返回大于阈值的像元个数
def tif_to_png_by_threshold(inputFile, saveFile, threshold, ignoreValue=0):
    raster_colorTable = gdal.ColorTable()
    # raster_colorTable.SetColorEntry(0, (255, 255, 255,0))
    raster_colorTable.SetColorEntry(1, (255, 0, 0, 255))
    logger.info('tif_to_png_by_threshold- inputFile is {}'.format(inputFile))
    DATA = get_data(inputFile, ignoreValue=ignoreValue)
    new_DATA = np.where(DATA >= threshold, 1, np.nan)
    area = np.nansum(new_DATA)

    temp_tif = saveFile.replace('png', 'tif')
    driver = gdal.GetDriverByName("GTIFF")
    dataset = driver.Create(temp_tif, DATA.shape[1], DATA.shape[0], 1, gdal.GDT_Byte)
    dataset.GetRasterBand(1).WriteArray(new_DATA)
    dataset.GetRasterBand(1).SetRasterColorTable(raster_colorTable)

    png_driver = gdal.GetDriverByName("PNG")
    logger.info('tif_to_png_by_threshold- saveFile is {}'.format(saveFile))
    png_driver.CreateCopy(saveFile, dataset)
    dataset = None
    png_driver = None
    os.remove(temp_tif)
    return area


# 使用指定的tif文件和色带生产渲染PNG图，GIS制图需要
# inputFile：输入tif文件
# saveFile：保存的PNG文件路径
# clevs ： 渲染分割数组：例如np.linspace(-10, 15, 6)，最小值为-10，最大值为15，分6类
# colorTable：颜色数组，格式为16进制
# ignoreValue：制图时忽略的tif中无效值
def tif_to_png(inputFile, saveFile, clevs, colorTable, ignoreValue=0):
    raster_colorTable = gdal.ColorTable()
    for i in range(len(colorTable) - 1):
        color_maxvalue = int((i + 1) * (255 / (len(colorTable) - 1)))
        if color_maxvalue > 255:
            color_maxvalue = 255
        raster_colorTable.CreateColorRamp(1 + int(i * (255 / (len(colorTable) - 1))), color(colorTable[i]),
                                          color_maxvalue, color(colorTable[i + 1]))
    raster_colorTable.SetColorEntry(0, (255, 255, 255, 0))

    DATA = get_data(inputFile, ignoreValue=ignoreValue)
    DATA_max = max(clevs)
    DATA_min = min(clevs)
    new_DATA = LineaStretch(DATA, DATA_min, DATA_max, bot=1)
    new_DATA[DATA == np.nan] = 0
    temp_tif = saveFile.replace('png', 'tif')
    driver = gdal.GetDriverByName("GTIFF")
    dataset = driver.Create(temp_tif, DATA.shape[1], DATA.shape[0], 1, gdal.GDT_Byte)
    dataset.GetRasterBand(1).WriteArray(new_DATA)
    dataset.GetRasterBand(1).SetRasterColorTable(raster_colorTable)

    pngdriver = gdal.GetDriverByName("PNG")
    pngdriver.CreateCopy(saveFile, dataset)
    dataset = None
    pngdriver = None
    os.remove(temp_tif)


# 按照给定的范围，动态调整tif数数值
# inputFile：输入tif文件
# saveFile：保存的tif文件路径
# min：动态调整范围
# max：动态调整范围
def make_similartif(inputFile, saveFile, min, max):
    input_dateset = gdal.Open(inputFile)
    width = input_dateset.RasterXSize
    height = input_dateset.RasterYSize
    geotrans = input_dateset.GetGeoTransform()
    proj = input_dateset.GetProjection()
    bands = input_dateset.RasterCount

    data = input_dateset.GetRasterBand(1).ReadAsArray()
    bandsdata = []
    for i in range(1, bands + 1):
        banddata = np.array(input_dateset.GetRasterBand(i).ReadAsArray())
        # 将无效值转化为Nan
        banddata = np.where(banddata != 0, banddata, np.nan)
        bandsdata.append(banddata)

    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create(saveFile, width, height, bands, gdal.GDT_Float32)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    for i, bd in zip(range(1, bands + 1), bandsdata):
        out_band = out_ds.GetRasterBand(i)
        rand = np.random.randint(0, 100, (bd.shape[0], bd.shape[1]))
        rand = np.interp(rand, (rand.min(), rand.max()), (min, max))
        bd = bd + rand
        out_band.WriteArray(bd)
    del out_ds


# 对数组进行线性拉伸，输出type为int8
# band：输入数组
# min：拉伸的范围最小值
# max：拉伸的范围最大值
# truncated_value：置信区间：例如：5：百分5%
# bot：拉伸后的bot最小值，如拉伸后的数组为0——255，比bot小的数组变为bot
# top：拉伸后的top最大值，如拉伸后的数组为0——255，比top大的数组变为top
def LineaStretch(band, min=None, max=None, truncated_value=None, bot=0, top=255):
    if truncated_value != None:
        min = np.nanpercentile(band, truncated_value)
        max = np.nanpercentile(band, 100 - truncated_value)
    band = 1.0 * (band - min) / (max - min) * (top - bot) + bot
    band[band < bot] = bot
    band[band > top] = top
    return band.astype(np.uint8)


def ScaleModis_ScaleIt(input, output):
    func = interpolate.interp1d(input, output, kind='linear')
    return func


# 真彩色制图
# redref：红波段数组
# greenref：绿波段数组
# blueref：波段数组
# saveFile：输出的PNG存储路径
def draw_RGB(redref, greenref, blueref, saveFile, cloud=None):
    if cloud == None:
        input = [0, 25, 55, 100, 255]
        output = [0, 90, 140, 175, 255]
    else:
        input = [0, 30, 60, 120, 190, 255]
        output = [0, 110, 160, 210, 240, 255]
    redstretch = LineaStretch(redref, truncated_value=2)
    # print(redstretch.max(),redstretch.min())
    greenstretch = LineaStretch(greenref, truncated_value=2)
    # print(greenstretch.max(), greenstretch.min())
    bluestretch = LineaStretch(blueref, truncated_value=2)
    # print(bluestretch.max(), bluestretch.min())
    height, width = redstretch.shape
    spline = ScaleModis_ScaleIt(input, output)
    redscale = spline(redstretch.flatten())
    # print(redscale.max(),redscale.min())
    greenscale = spline(greenstretch.flatten())
    # print(greenscale.max(), greenscale.min())
    bluescale = spline(bluestretch.flatten())
    # print(bluescale.max(), bluescale.min())
    ascale = np.where(np.isnan(redscale), 0, 255)
    result = np.dstack((redscale, greenscale, bluescale, ascale)).reshape(height, width, 4)
    new_im = Image.fromarray(result.astype(np.uint8)).convert('RGBA')
    new_im.save(saveFile, format='png')
    return result


# 使用命令行调用seadas将Modis L1A级数据处理为L2级
# l1filepath：L1A级数组路径
# l2dir：L2A级数组存储路径
# done：做完的数组存储路径
def MODIS_L1A_LAC_TO_L2_LAC(l1filepath, l2dir, donedir):
    base = os.path.splitext(l1filepath)
    geo_file = base[0] + '.GEO'
    l1b_file = base[0] + '.L1B_LAC'
    l2_file = os.path.join(l2dir, os.path.basename(base[0]) + '.L2_LAC')
    anc_file = base[0] + '.L1A_LAC.anc'
    os.system('cd ' + os.path.dirname(l1filepath) + ' && getanc.py  -v  ' + l1filepath)
    os.system('modis_GEO.py -v ' + l1filepath + ' -o ' + geo_file)
    os.system('modis_L1B.py -v ' + l1filepath + ' -o ' + l1b_file)
    os.system('l2gen ifile=' + l1b_file + ' geofile=' + geo_file + ' ofile=' + l2_file +
              ' par=' + anc_file + ' l2prod="chlor_a rhos_nnn"')
    os.remove(geo_file)
    os.remove(anc_file)
    L1B_files = glob.glob(base[0] + '.L1B*')
    for L1B_file in L1B_files:
        os.remove(L1B_file)
    os.system('mv -v ' + l1filepath + ' ' + os.path.join(donedir, os.path.basename(l1filepath)))
    return l2_file


# 使用命令行调用seadas将GOCI L1A级数据处理为L2级
# l1filepath：L1A级数组路径
# l2dir：L2A级数组存储路径
# done：做完的数组存储路径
def GOCI_L1A_LAC_TO_L2_LAC(l1filepath, l2dir, donedir):
    base = os.path.splitext(l1filepath)
    l2_file = os.path.join(l2dir, os.path.basename(base[0]) + '.L2_LAC')
    anc_file = base[0] + '.he5.anc'
    os.system('cd ' + os.path.dirname(l1filepath) + ' && getanc.py  -v  ' + l1filepath)
    os.system('l2gen ifile=' + l1filepath + ' ofile=' + l2_file +
              ' l2prod="chlor_a rhos_nnn"')
    os.remove(anc_file)
    os.system('mv -v ' + l1filepath + ' ' + os.path.join(donedir, os.path.basename(l1filepath)))
    return l2_file


def FAI_hu2009(RED: np.ndarray, L_RED: int, NIR: np.ndarray, L_NIR: int, SWIR: np.ndarray, L_SWIR: int, saveFile,
               profile):
    R_rc_s = RED + (SWIR - RED) * (L_NIR - L_RED) / (L_SWIR - L_RED)
    FAI = NIR - R_rc_s

    gdal_array.numpy.seterr(all="ignore")
    out = gdal_array.SaveArray(FAI, saveFile, format="GTiff", prototype=profile)
    out = None


def AFAI_Qi2018(RED: np.ndarray, L_RED: int, NIR1: np.ndarray, L_NIR1: int, NIR2: np.ndarray, L_NIR2: int, saveFile,
                profile):
    R_rc_s = RED + (NIR2 - RED) * (L_NIR1 - L_RED) / (L_NIR2 - L_RED)
    FAI = NIR1 - R_rc_s
    gdal_array.numpy.seterr(all="ignore")
    out = gdal_array.SaveArray(FAI, saveFile, format="GTiff", prototype=profile)
    out = None

# if __name__ == '__main__':
#     inputFiles = [r'D:\taihulanzao\result\GK2A\GK2A_taihu_202109171230_FAI_subset.tif',
#                   r'D:\taihulanzao\result\GOCI\GOCI_taihu_201711231416_FAI_subset.tif',
#                   r'D:\taihulanzao\result\MODIS\Modis_taihu_201711221340_FAI_subset.tif']
#     saveFile = r'D:\taihulanzao\layer_stack.tif'
#     layer_stack(inputFiles,saveFile)

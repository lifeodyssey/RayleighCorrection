import pandas as pd
import netCDF4 as nc4
import numpy as np

from geo_Collection import geo_self as gs
import cartopy.crs as ccrs
import os
import glob
import datetime
import warnings
import multiprocessing as mp
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from osgeo import ogr
from osgeo import osr
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from osgeo import gdal

from base import MeanDEM# 这个暂时没用到 后续可以优化

from datetime import date
from base import PredefinedWavelength
from Py6S import *

path='/root/Data/H8/20210917/'
os.chdir('/root/Data/H8/20210917')
datalist = glob.glob('AHI8*2000*.hdf')
LONLAT=nc4.Dataset('/root/taihu/h8ra/AHI8_OBI_2000M_NOM_LATLON.HDF', 'r')
latmax=31.555000       
lonmin=119.89000
latmin=30.916291       
lonmax=120.57894

lat=np.array(LONLAT['Lat'])
lon=np.array(LONLAT['Lon'])

SatAzi=nc4.Dataset('/root/taihu/h8ra/AHI8_OBI_2000M_NOM_SATAZI.HDF', 'r')['SatAzimuth']
SatZen=nc4.Dataset('/root/taihu/h8ra/AHI8_OBI_2000M_NOM_SATZEN.HDF', 'r')['SatZenith']
geotrans = (lonmin, 0.02, 0, latmax, 0, 0.02)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
proj = srs.ExportToWkt()

def calParameter(resolution):
    '''
    @desc: 计算Himawari-8 COFF/CFAC/LOFF/LFAC
    @resolution: 10:1km/20:2km
    '''
    if resolution == 20:
        row = 550
        col = 5500
        COFF = LOFF = 2750.5
        CFAC = LFAC = 20466275
    elif resolution == 10:
        row = 1100
        col = 11000
        COFF = LOFF = 5500.5
        CFAC = LFAC = 40932549
    return COFF, LOFF, CFAC, LFAC
def write_geotiff(filename, arr, Projection, Transformation):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
	# 确定是用arr本身的数据类型还是其他的数据类型
    driver = gdal.GetDriverByName("GTiff")
    # 确定输出数据类型
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    # 确定一些输出数据的参数，包括文件名，column和height，多少个波段(那个1)，什么数据类型
    out_ds.SetProjection(Projection)
    out_ds.SetGeoTransform(Transformation)
    # 确定数据的投影和坐标系，这里他用的是原本的那个
    # 除此之外，还可以使用绝对的坐标系
    # GeoTransform 的形式为 (486892.5, 15.0, 0.0, 4105507.5, 0.0, -15.0)
    # 六个参数分别为 左上角x坐标， 水平分辨率，旋转参数， 左上角y坐标，旋转参数，竖直分辨率
    # 一般旋转参数都设为0
    # SetProjection则可以是OGC WKT或者PROJ.4格式的字符串 可以从https://cfconventions.org/wkt-proj-4.html查到
    # 
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

def latlon2linecolumn(lat, lon, resolution):
    """
    经纬度转行列
    (lat, lon) → (line, column)
    resolution：文件名中的分辨率
    line, column
    """
    COFF, LOFF, CFAC, LFAC = calParameter(resolution)
    ea = 6378.137  # 地球的半长轴[km]
    eb = 6356.7523  # 地球的短半轴[km]
    h = 42164  # 地心到卫星质心的距离[km]
    λD = np.deg2rad(140.7)  # 卫星星下点所在经度
    # Step1.检查地理经纬度
    # Step2.将地理经纬度的角度表示转化为弧度表示
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    # Step3.
    #将地理经纬度转化成地心经纬度
    eb2_ea2 = eb ** 2 / ea ** 2
    λe = lon
    φe = np.arctan(eb2_ea2 * np.tan(lat))
    # Step4.求Re
    cosφe = np.cos(φe)
    re = eb / np.sqrt(1 - (1 - eb2_ea2) * cosφe ** 2)
    # Step5.求r1,r2,r3
    λe_λD = λe - λD
    r1 = h - re * cosφe * np.cos(λe_λD)
    r2 = -re * cosφe * np.sin(λe_λD)
    r3 = re * np.sin(φe)
    # Step6.求rn,x,y
    rn = np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    x = np.rad2deg(np.arctan(-r2 / r1))
    y = np.rad2deg(np.arcsin(-r3 / rn))
    # Step7.求c,l
    column = COFF + x * 2 ** -16 * CFAC
    line = LOFF + y * 2 ** -16 * LFAC
    return np.rint(line).astype(np.uint16), np.rint(column).astype(np.uint16)
def hsd2tif(arr, lonMin, lonMax, latMin, latMax, pixelSize,resolution,file_out:str=None):
    '''
    @desc: 几何校正，输出tif
    '''
    col = np.ceil(lonMax-lonMin)/pixelSize
    row = np.ceil(latMax-latMin)/pixelSize
    col = int(col)
    row = int(row)
    ynew = np.linspace(latMax, latMin, row)
    xnew = np.linspace(lonMin, lonMax, col)
    xnew,ynew = np.meshgrid(xnew, ynew)
    dataGrid = np.zeros((col,row)) 
    
    index = {}
    for i in range(row):
        for j in range(col):
            lat = ynew[j][i]
            lon = xnew[j][i]
            h8Row = 0
            h8Col = 0
            if index.get((lat, lon)) == None:
                h8Row, h8Col = latlon2linecolumn(lat, lon, resolution)
                index[(lat, lon)] = h8Row, h8Col
            else:
                h8Row, h8Col = index.get((lat, lon))
                
            if (0<=h8Row<11000) and (0<=h8Col<11000):
                dataGrid[j][i] = arr[h8Row,h8Col,]
            else:
                print("该坐标(%s, %s)不在影像中"%(lon, lat))
    #[119.5,0.005,0,32,0,0.005]
    geotrans = (lonMin, pixelSize, 0, latMax, 0, pixelSize)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    proj = srs.ExportToWkt()
    #filename, arr, Projection, Transformation
    if file_out is not None:
        write_geotiff(file_out,dataGrid,proj,geotrans)
    return dataGrid
def RayleighCorrection_h8(filename, R_toa,band,SolarZenith,SolarAzimuth,SatelliteZenith,SatelliteAzimuth,latmax,latmin,lonmax,lonmin):
    # 后期可以优化一下，这里只能一个点一个点的算，有时间了可以改成开并行
    assert np.shape(R_toa)==np.shape(SolarZenith)
    R=np.ones_like(R_toa)
    Lonn,Latn=np.shape(R_toa)
    Lon_step=(lonmax-lonmin)/Lonn
    Lat_step=(latmax-latmin)/Latn
    month = int(filename[23:25])
    day = int(filename[25:27])
    #这里开并行
    for i in range(Lonn):
        for j in range(Latn):
    # 6S模型
            s = SixS()

            # 传感器类型 自定义,
            s.geometry = Geometry.User()
            s.geometry.solar_z = SolarZenith[i,j]
            s.geometry.solar_a = SolarAzimuth[i,j]
            s.geometry.view_z = SatelliteZenith[i,j]
            s.geometry.view_a = SatelliteAzimuth[i,j]
        # 日期
            s.geometry.month = month
            s.geometry.day = day
        #s.geometry.month = 9
        #s.geometry.day = 17
        # print(s.geometry)
        # 中心经纬度
            TopLeftLat = latmax-j*Lat_step+0.5*Lat_step
            TopLeftLon = lonmin+(i)*Lon_step-0.5*Lon_step
            TopRightLat = latmax-j*Lat_step+0.5*Lat_step
            TopRightLon = lonmin+(i)*Lon_step+0.5*Lon_step
            BottomRightLat = latmax-(j)*Lat_step-0.5*Lat_step
            BottomRightLon = lonmin+(i)*Lon_step+0.5*Lon_step
            BottomLeftLat = latmax-(j)*Lat_step-0.5*Lat_step
            BottomLeftLon = lonmin+(i)*Lon_step-0.5*Lon_step

            ImageCenterLat = (TopLeftLat + TopRightLat + BottomRightLat + BottomLeftLat) / 4

        # 大气模式类型
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)

        # 气溶胶类型大陆
            s.aero_profile = AtmosProfile.PredefinedType(AeroProfile.NoAerosols)

        # 下垫面类型
            s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0)


        # # 通过研究去区的范围去求DEM高度。
        #     pointUL = dict()
        #     pointDR = dict()
        #     pointUL["lat"] = max(TopLeftLat,TopRightLat,BottomRightLat,BottomLeftLat)
        #     pointUL["lon"] = min(TopLeftLon,TopRightLon,BottomRightLon,BottomLeftLon)
        #     pointDR["lat"] = min(TopLeftLat,TopRightLat,BottomRightLat,BottomLeftLat)
        #     pointDR["lon"] = max(TopLeftLon,TopRightLon,BottomRightLon,BottomLeftLon)
        #     meanDEM = (MeanDEM(pointUL, pointDR)) * 0.001

        # 研究区海拔、卫星传感器轨道高度
            s.altitudes = Altitudes()
            s.altitudes.set_target_custom_altitude(0)
            s.altitudes.set_sensor_satellite_level()

        # 校正波段（根据波段名称）
            if band == 640:
                SRFband = PredefinedWavelength.Himawari_640[3]
                wvmin=PredefinedWavelength.Himawari_640[1]
                wvmax=PredefinedWavelength.Himawari_640[2]
                s.wavelength = Wavelength(wvmin,wvmax,SRFband)

            elif band == 860:
                SRFband = PredefinedWavelength.Himawari_860[3]
                wvmin=PredefinedWavelength.Himawari_860[1]
                wvmax=PredefinedWavelength.Himawari_860[2]
                s.wavelength = Wavelength(wvmin,wvmax,SRFband)

            elif band == 1600:
                SRFband = PredefinedWavelength.Himawari_1600[3]
                wvmin=PredefinedWavelength.Himawari_1600[1]
                wvmax=PredefinedWavelength.Himawari_1600[2]
                s.wavelength = Wavelength(wvmin,wvmax,SRFband)



        # 运行6s大气模型
            s.run()
            Rr=s.outputs.apparent_reflectance
            R[i,j]=Rr
    Rrc=R_toa-R
    # x = s.outputs.values
    return Rrc
def FAI_hu2009(RED:np.ndarray,L_RED:int,NIR:np.ndarray,L_NIR:int,SWIR:np.ndarray,L_SWIR:int):
    R_rc_s=RED+(SWIR-RED)*(L_NIR-L_RED)/(L_SWIR-L_RED)
    FAI=NIR-R_rc_s
    return FAI
import datetime  


for f in datalist:
    # 可以直接在这里把高于70°太阳高度角的给去掉
    starttime = datetime.datetime.now ()
    print('Processing')
    print(f)
    nc_file = nc4.Dataset(f, 'r')
    date1=f[19:27]
    t=f[-8:-4]
    ID640=np.asarray(nc_file.variables['NOMChannelVIS0046_2000'])
    ID860=np.asarray(nc_file.variables['NOMChannelVIS0086_2000'])
    ID1600=np.asarray(nc_file.variables['NOMChannelVIS0160_2000'])
    
    R640=np.ones_like(ID640,dtype='float')
    R860=np.ones_like(ID860,dtype='float')    
    R1600=np.ones_like(ID1600,dtype='float')

    R640_value=np.array(nc_file.variables['Radiance_0064'])
    R860_value=np.array(nc_file.variables['Radiance_0086'])
    R1600_value=np.array(nc_file.variables['Radiance_0160'])
    
    for i in range(5500):
        for j in range(5500):
            R640[i,j]=R640_value[ID640[i,j]-1]
            R860[i,j]=R860_value[ID860[i,j]-1]
            R1600[i,j]=R1600_value[ID1600[i,j]-1]# 这个循环可以优化一下
    
        SatAzi=nc4.Dataset('/root/taihu/h8ra/AHI8_OBI_2000M_NOM_SATAZI.HDF', 'r')['SatAzimuth']
    SatZen=nc4.Dataset('/root/taihu/h8ra/AHI8_OBI_2000M_NOM_SATZEN.HDF', 'r')['SatZenith']
    SatZen_Sub=hsd2tif(SatZen,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02,resolution=20)
    SatAzi_Sub=hsd2tif(SatAzi,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02,resolution=20)
    
    
    R640_sub=hsd2tif(R640,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02, file_out=date1+t+'R640.tif',resolution=20)
    R860_sub=hsd2tif(R860,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02, file_out=date1+t+'R860.tif',resolution=20)
    R1600_sub=hsd2tif(R1600,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02, file_out=date1+t+'R1600.tif',resolution=20)
    SunZenith=nc_file.variables['NOMSunZenith']
    SunAzi=nc_file.variables['NOMSunAzimuth']
    SunZen_Sub=hsd2tif(SunZenith,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02,resolution=20)
    SunAzi_Sub=hsd2tif(SunAzi,lonMin=lonmin, lonMax=lonmax, latMin=latmin, latMax=latmax, pixelSize=0.02,resolution=20)

    cos_theta=np.cos(SunZen_Sub)
    R640_toa=np.pi*R640_sub/(1631.5726*cos_theta)
    R860_toa=np.pi*R860_sub/(971.8778*cos_theta)
    R1600_toa=np.pi*R1600_sub/(242.3462*cos_theta)
    
    Rrc640=RayleighCorrection_h8(R_toa=R640_toa,filename=f,
                             band=640,
                             SolarZenith=SunZen_Sub,
                             SolarAzimuth=SunAzi_Sub,
                             SatelliteZenith=SatZen_Sub,
                             SatelliteAzimuth=SatAzi_Sub,
                            latmax=latmax,
                            latmin=latmin,
                            lonmax=lonmax,
                            lonmin=lonmin)
    Rrc860=RayleighCorrection_h8(R_toa=R860_toa,filename=f,
                             band=860,
                             SolarZenith=SunZen_Sub,
                             SolarAzimuth=SunAzi_Sub,
                             SatelliteZenith=SatZen_Sub,
                             SatelliteAzimuth=SatAzi_Sub,
                            latmax=latmax,
                            latmin=latmin,
                            lonmax=lonmax,
                            lonmin=lonmin)
    Rrc1600=RayleighCorrection_h8(R_toa=R1600_toa,filename=f,
                             band=1600,
                             SolarZenith=SunZen_Sub,
                             SolarAzimuth=SunAzi_Sub,
                             SatelliteZenith=SatZen_Sub,
                             SatelliteAzimuth=SatAzi_Sub,
                            latmax=latmax,
                            latmin=latmin,
                            lonmax=lonmax,
                            lonmin=lonmin)
    FAI=FAI_hu2009(RED=Rrc640,L_RED=640,
              NIR=Rrc860,L_NIR=860,
              SWIR=Rrc1600,L_SWIR=1600)
    write_geotiff('MODDIS_taihu_'+date1+t+'.tif', FAI,proj,geotrans )
    endtime = datetime.datetime.now () 
    print ((endtime - starttime).seconds)
    print('Finished,Next')
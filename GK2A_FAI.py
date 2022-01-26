import pandas as pd
import netCDF4 as nc
import numpy as np

import os
import glob
import datetime
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import ogr
from osgeo import osr
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from osgeo import gdal
import pysolar
from datetime import timezone
from base import MeanDEM
from base import PredefinedWavelength
from Py6S import *
import cv2


ea = 6378.137          # 地球长半轴
eb = 6356.7523         # 地球短半轴
H = 42164.
TWOPI = 6.28318530717958648 #不知道干嘛的参数
DPAI = 6.28318530717958648#不知道干嘛的参数
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi
FILLVALUE = -999
FLAT = 0.00335281317789691#不知道干嘛的参数
AE = 6378.137#不知道干嘛的参数
E2 = 0.0066943800699785#不知道干嘛的参数

left_upper_lat=31.555000 
left_upper_lon=119.89000
right_lower_lat=30.916291 
right_lower_lon=120.57894

latmax=left_upper_lat     
lonmin=left_upper_lon
latmin=right_lower_lat      
lonmax=right_lower_lon
geotrans = (lonmin, 0.02, 0, latmax, 0, 0.02)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
proj = srs.ExportToWkt()



def unzip(fname, dirs):
    try:
        f = zipfile.ZipFile(fname)
        f.extractall(path=dirs)
        f.close()
    except Exception as e:
        print("文件%s打开失败" % fname)
        
def latlon_from_lincol_geos(Resolution, Line ,Column):
	degtorad=3.14159265358979 / 180.0
	if(Resolution == 0.5):
		COFF=11000.5
		CFAC=8.170135561335742e7
		LOFF=11000.5
		LFAC=8.170135561335742e7
	elif(Resolution == 1.0):
		COFF=5500.5
		CFAC=4.0850677806678705e7
		LOFF=5500.5
		LFAC=4.0850677806678705e7
	else:
		COFF=2750.5
		CFAC=2.0425338903339352e7
		LOFF=2750.5
		LFAC=2.0425338903339352e7
	sub_lon=128.2
	sub_lon=sub_lon*degtorad
	
	x= degtorad *( (Column - COFF)*2**16 / CFAC )
	y= degtorad *( (Line - LOFF)*2**16 / LFAC )
	Sd = np.sqrt( (42164.0*np.cos(x)*np.cos(y))**2 - (np.cos(y)**2 + 1.006739501*np.sin(y)**2)*1737122264)
	Sn = (42164.0*np.cos(x)*np.cos(y)-Sd) / (np.cos(y)**2 + 1.006739501*np.sin(y)**2)
	S1 = 42164.0 - ( Sn * np.cos(x) * np.cos(y) )
	S2 = Sn * ( np.sin(x) * np.cos(y) )
	S3 = -Sn * np.sin(y)
	Sxy = np.sqrt( ((S1*S1)+(S2*S2)) )
	
	
	nlon=(np.arctan(S2/S1)+sub_lon)/degtorad
	nlat=np.arctan( ( 1.006739501 *S3)/Sxy)/degtorad
	
	return (nlat, nlon)


##########################
# 3. Define function : (Latitude, Longitude)	->	Full disc GEOS image(Line, Column)
##########################
def lincol_from_latlon_geos(Resolution, Latitude, Longitude):
	degtorad=3.14159265358979 / 180.0
	if(Resolution == 0.5):
		COFF=11000.5
		CFAC=8.170135561335742e7
		LOFF=11000.5
		LFAC=8.170135561335742e7
	elif(Resolution == 1.0):
		COFF=5500.5
		CFAC=4.0850677806678705e7
		LOFF=5500.5
		LFAC=4.0850677806678705e7
	else:
		COFF=2750.5
		CFAC=2.0425338903339352e7
		LOFF=2750.5
		LFAC=2.0425338903339352e7
	
	sub_lon=128.2
	sub_lon=sub_lon*degtorad
	Latitude=Latitude*degtorad
	Longitude=Longitude*degtorad
	
	c_lat = np.arctan(0.993305616*np.tan(Latitude))
	RL =  6356.7523 / np.sqrt( 1.0 - 0.00669438444*np.cos(c_lat)**2.0 )
	R1 =  42164.0 - RL *np.cos(c_lat)*np.cos(Longitude - sub_lon)
	R2 = -RL* np.cos(c_lat) *np.sin(Longitude - sub_lon)
	R3 =  RL* np.sin(c_lat)
	Rn =  np.sqrt(R1**2.0 + R2**2.0 + R3**2.0 )
	
	x = np.arctan(-R2 / R1) / degtorad
	y = np.arcsin(-R3 / Rn) / degtorad
	ncol=COFF + (x* 2.0**(-16) * CFAC)
	nlin=LOFF + (y* 2.0**(-16) * LFAC)
	return (nlin,ncol)


def cut_with_latlon_geos(Array, Resolution, Latitude1, Longitude1, Latitude2, Longitude2):
	Array=np.array(Array)
	if(Resolution == 0.5):
		Index_max=22000
	elif(Resolution == 1.0):
		Index_max=11000
	else:
		Index_max=5500
	
	(Lin1,Col1) = lincol_from_latlon_geos(Resolution, Latitude1, Longitude1)
	(Lin2,Col2) = lincol_from_latlon_geos(Resolution, Latitude2, Longitude2)
	Col1=int(np.floor(Col1))
	Lin1=int(np.floor(Lin1))
	Col2=int(np.ceil(Col2))
	Lin2=int(np.ceil(Lin2))
	
	cut=np.zeros((Index_max,Index_max))
	if( (Col1 <= Col2) and (Lin1 <= Lin2) and (0 <= Col1) and (Col2 < Index_max) and (0 <= Lin1) and (Lin2 < Index_max) ):
		cut=Array[Lin1:Lin2,Col1:Col2]
	
	return cut


def Read_Radiance(fname):

    # Cut and calculate Radiance
    ############################
    # 6. GK2A sample data file read
    ############################
    input_ncfile = nc.Dataset(fname,'r',format='netcdf4')

    ipixel=input_ncfile.variables['image_pixel_values']
    sr=float(input_ncfile.channel_spatial_resolution)

    ##########################
    # 7. Calculate latitude & longitude from GEOS image
    ##########################
    i = np.arange(0,input_ncfile.getncattr('number_of_columns'),dtype='f')
    j = np.arange(0,input_ncfile.getncattr('number_of_lines'),dtype='f')
    i,j = np.meshgrid(i,j)

    (geos_lat,geos_lon) = latlon_from_lincol_geos(sr,j,i)


    ##########################
    # 8. Cut user defined area from GEOS image
    ##########################
    cut_pixel=cut_with_latlon_geos(ipixel[:],sr,left_upper_lat,left_upper_lon,right_lower_lat,right_lower_lon)
    cut_lat=cut_with_latlon_geos(geos_lat,sr,left_upper_lat,left_upper_lon,right_lower_lat,right_lower_lon)
    cut_lon=cut_with_latlon_geos(geos_lon,sr,left_upper_lat,left_upper_lon,right_lower_lat,right_lower_lon)

    (ulc_lin,ulc_col)=lincol_from_latlon_geos(sr,left_upper_lat,left_upper_lon)
    (lrc_lin,lrc_col)=lincol_from_latlon_geos(sr,right_lower_lat,right_lower_lon)


    ############################
    # 9. image_pixel_values DQF processing
    ############################
    cut_pixel[cut_pixel>49151] = 0 #set error pixel's value to 0

    ############################
    #10. image_pixel_values Bit Size per pixel masking
        ############################
    channel=ipixel.getncattr('channel_name')
    if ((channel == 'VI004') or (channel == 'VI005') or (channel == 'NR016')):
        mask = 0b0000011111111111 #11bit mask
    elif ((channel == 'VI006') or (channel == 'NR013') or (channel == 'WV063')):
        mask = 0b0000111111111111 #12bit mask
    elif (channel == 'SW038'):
        mask = 0b0011111111111111 #14bit mask
    else:
        mask = 0b0001111111111111 #13bit mask

    cut_pixel_masked=np.bitwise_and(cut_pixel,mask)


    ############################
    # 11. image pixel value -> Radiance
    ############################

    Gain=input_ncfile.DN_to_Radiance_Gain
    Offset=input_ncfile.DN_to_Radiance_Offset
  
    R=cut_pixel_masked*Gain+Offset
    input_ncfile.close()
    return (np.array(R),cut_lat,cut_lon)


def _Zocgef( lat, lon, height):
    lat = np.array(lat)
    lon = np.array(lon)

    Xge = [] # np.zeros(3, dtype=np.float64)
    E = FLAT * (2.0 - FLAT)
    ENP = (AE * 1000) / np.sqrt(1.0 - E * np.sin(lat * deg2rad) * E * np.sin(lat * deg2rad))
    Xge.append( (ENP + height) * np.cos(lat * deg2rad) * np.cos(lon * deg2rad) / 1000.)


    Xge.append((ENP + height) * np.cos(lat * deg2rad) * np.sin(lon * deg2rad) / 1000.)

    Xge.append((ENP * (1.0 - E) + height) * np.sin(lat * deg2rad) / 1000.)

    return Xge
def _ZSODSRadto2Pai(dRad) :

    dresult = dRad - DPAI*((int)(dRad/DPAI))

    if dRad < 0.0 :
        dresult += DPAI

    return dresult

def _ICGS(RX, Sth, Cth, lat):
    HGT = 50.0 / 6378137.0
    Sla = np.sin(lat * deg2rad)
    Cla = np.cos(lat * deg2rad)
    RE2 = AE / np.sqrt(1.0 -E2 * Sla * Sla)
    Gxic = (RE2 + HGT) * Cla * Cth

    Gyic = (RE2 + HGT) * Cla * Sth
    Gzic = (RE2 * (1.0 - E2) + HGT) * Sla
    Rhx = RX[0] - Gxic
    Rhy = RX[1] - Gyic
    Rhz = RX[2] - Gzic
    Rht = Cth * Rhx + Sth * Rhy
    Rxgs = Cth * Rhy - Sth * Rhx
    Rygs = Cla * Rhz - Sla * Rht
    Rzgs = Cla * Rht + Sla * Rhz

    Azgs = np.arctan2(Rxgs, Rygs)

    Azgs[Azgs < 0.0] = Azgs[Azgs < 0.0] + TWOPI
    Elgs = np.arctan(Rzgs / np.sqrt(Rxgs * Rxgs + Rygs * Rygs))

    return Azgs, Elgs
def _ICGSSAT( lat, lon, RSat, RX):

    HGT = 50.0 / 6378137.0
    Sla = np.sin(lat * deg2rad)
    Cla = np.cos(lat * deg2rad)
    Sth = np.sin(lon * deg2rad)
    Cth = np.cos(lon * deg2rad)

    Rhx = RSat[0] - RX[0]
    Rhy = RSat[1] - RX[1]
    Rhz = RSat[2] - RX[2]
    Rht = Cth * Rhx + Sth * Rhy
    Rxgs = Cth * Rhy - Sth * Rhx
    Rygs = Cla * Rhz - Sla * Rht
    Rzgs = Cla * Rht + Sla * Rhz

    Azgs = np.arctan2(Rxgs, Rygs)
    Azgs[Azgs < 0.0] = Azgs[Azgs < 0.0] + TWOPI
    Azgs[Azgs < np.pi] = Azgs[Azgs < np.pi] + np.pi
    Azgs[Azgs >= np.pi] = Azgs[Azgs >= np.pi] - np.pi

    Elgs = np.arctan(Rzgs / np.sqrt(Rxgs * Rxgs + Rygs * Rygs))

    return Elgs



def CalSatZenith(lat, lon, subpoint = 128.2) :
    lat = np.array(lat, dtype=np.float32)
    lon = np.array(lon, dtype=np.float32)
    fillflag = (lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)

    c_lat = np.arctan(0.993243 * np.tan(lat * deg2rad))
    rl = 6356.7523 / np.sqrt(1 - 0.00675701 * np.cos(c_lat) * np.cos(c_lat))
    r1 = H - rl * np.cos(c_lat) * np.cos(lon * deg2rad - subpoint * deg2rad)
    r2 = -rl * np.cos(c_lat) * np.sin(lon * deg2rad - subpoint * deg2rad)

    r3 = rl * np.sin(c_lat)
    rn = np.sqrt(r1 * r1 + r2 * r2 + r3 * r3)

    r4 = np.arccos((rn * rn + H * H - rl * rl) / (2 * rn * H))

    r5 = np.arcsin(np.sin(r4) * H / 6356.7523) * rad2deg

    r5[fillflag] = FILLVALUE

    return r5


def CalSatAzimuth( lat, lon, subpoint = 128.2):
    '''
    计算卫星方位角
    :param lat: array_like,
                unit: degree
    :param lon: array_like,
                unit: degree
    :param subpoint: float,
                unit: degree
    :return: satellite azimuth , array_like:
                unit:degree
    '''
    lat = np.array(lat, dtype=np.float32)
    lon = np.array(lon, dtype=np.float32)
    fillflag = (lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)

    heigh = H * 1.0E3


    RX = _Zocgef(lat, lon, heigh)

    RSat = _Zocgef(0.0, subpoint, heigh)

    Azi = _ICGSSAT(lat, lon, RSat, RX)

    sata = Azi * rad2deg

    flag = sata >= 180
    sata[flag] -= 180.0
    sata[~flag] += 180.0

    sata[fillflag] = FILLVALUE

    return sata


def RayleighCorrection_GK2A(Time, R_toa,band,SolarZenith,SolarAzimuth,SatelliteZenith,SatelliteAzimuth,latmax,latmin,lonmax,lonmin):
    # 后期可以优化一下，这里只能一个点一个点的算，有时间了可以改成开并行
    assert np.shape(R_toa)==np.shape(SolarZenith)
    R=np.ones_like(R_toa)
    Latn,Lonn=np.shape(R_toa)
    Lon_step=(lonmax-lonmin)/Lonn
    Lat_step=(latmax-latmin)/Latn
    month =Time .month
    day = Time.day
    for i in range(Latn):
        for j in range(Lonn):
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
                SRFband = PredefinedWavelength.Himawari_640[3]## 查了查 6S模型在某些波段跑不出来结果 用GK2A的波段函数的时候结果是nan，所以用了H8的
                wvmin=PredefinedWavelength.Himawari_640[1]# 这两个640的函数基本一样
                wvmax=PredefinedWavelength.Himawari_640[2]
                s.wavelength = Wavelength(wvmin,wvmax,SRFband)

            elif band == 856:
                SRFband = PredefinedWavelength.GK2A_856[3]
                wvmin=PredefinedWavelength.GK2A_856[1]
                wvmax=PredefinedWavelength.GK2A_856[2]
                s.wavelength = Wavelength(wvmin,wvmax,SRFband)

            elif band == 1600:
                SRFband = PredefinedWavelength.GK2A_1600[3]
                wvmin=PredefinedWavelength.GK2A_1600[1]
                wvmax=PredefinedWavelength.GK2A_1600[2]
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


def write_geotiff(filename, arr, Projection, Transformation):
    if (arr.dtype == np.float32) | (arr.dtype == np.float64):
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

ob_path='/root/Data/GK2A'
path='/root/Data/GK2A/'
os.chdir(ob_path)
starttime = datetime.datetime.now ()
print('start')
#输入文件是解压之后的
#记得加入sunzen 70的判断
nc640 = glob.glob('gk2a*vi006*0430*.nc')[0]
nc856 = glob.glob('gk2a*vi008*0430*.nc')[0]
nc1600 = glob.glob('gk2a*nr016*0430*.nc')[0]

R640=Read_Radiance(nc640)[0]

R856=Read_Radiance(nc856)[0]
R1600,sublat,sublon=Read_Radiance(nc1600)
print('Processing')
T= datetime.datetime.strptime('%s' % (nc640[-15:-3]), '%Y%m%d%H%M')
T=T.replace(tzinfo=datetime.timezone.utc)
print(T)
alt = pysolar.solar.get_altitude(sublat, sublon, T)
SunZen=90-alt
SunAzi=pysolar.solar.get_azimuth(sublat, sublon, T)
cos_theta=np.cos(SunZen*deg2rad)

SatZen=CalSatZenith(sublat,sublon)
SatAzi=CalSatAzimuth(sublat,sublon)

yn,xn=np.shape(SunZen)

dim=(xn,yn)
R6402=cv2.resize(R640, dsize=dim,interpolation= cv2.INTER_CUBIC)
R8562=cv2.resize(R856, dsize=dim,interpolation= cv2.INTER_CUBIC)

R640_toa=np.pi*R6402/(1638.95*cos_theta)
R856_toa=np.pi*R8562/(977.48*cos_theta)
R1600_toa=np.pi*R1600/(246.16*cos_theta)
endtime = datetime.datetime.now() 
print('time of cut and resample')
print (((endtime - starttime).seconds)/60)
Rrc640=RayleighCorrection_GK2A(R_toa=R640_toa,Time=T,
                             band=640,
                             SolarZenith=SunZen,
                             SolarAzimuth=SunAzi,
                             SatelliteZenith=SatZen,
                             SatelliteAzimuth=SatAzi,
                            latmax=latmax,
                            latmin=latmin,
                            lonmax=lonmax,
                            lonmin=lonmin)

Rrc856=RayleighCorrection_GK2A(R_toa=R856_toa,Time=T,
                             band=856,
                             SolarZenith=SunZen,
                             SolarAzimuth=SunAzi,
                             SatelliteZenith=SatZen,
                             SatelliteAzimuth=SatAzi,
                            latmax=latmax,
                            latmin=latmin,
                            lonmax=lonmax,
                            lonmin=lonmin)


Rrc1600=RayleighCorrection_GK2A(R_toa=R1600_toa,Time=T,
                             band=1600,
                             SolarZenith=SunZen,
                             SolarAzimuth=SunAzi,
                             SatelliteZenith=SatZen,
                             SatelliteAzimuth=SatAzi,
                            latmax=latmax,
                            latmin=latmin,
                            lonmax=lonmax,
                            lonmin=lonmin)


FAI=FAI_hu2009(RED=Rrc640,L_RED=640,
              NIR=Rrc856,L_NIR=860,
              SWIR=Rrc1600,L_SWIR=1600)
write_geotiff('GK2A_taihu_'+nc640[-15:-3]+'.tif', FAI,proj,geotrans )
endtime = datetime.datetime.now() 
print('total time')
print (((endtime - starttime).seconds)/60)
print('Finished,Next')

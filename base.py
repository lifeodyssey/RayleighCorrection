
import os
from osgeo import gdal
import numpy as np

def MeanDEM(pointUL, pointDR):
    '''
    计算影像所在区域的平均高程.
    '''
    script_path = os.path.split(os.path.realpath(__file__))[0]
    dem_path = os.path.join(script_path,"GMTED2km.tif")

    try:
        DEMIDataSet = gdal.Open(dem_path)
    except Exception as e:
        pass

    DEMBand = DEMIDataSet.GetRasterBand(1)
    geotransform = DEMIDataSet.GetGeoTransform()
    # DEM分辨率
    pixelWidth = geotransform[1]
    pixelHight = geotransform[5]

    # DEM起始点：左上角，X：经度，Y：纬度
    originX = geotransform[0]
    originY = geotransform[3]

    # 研究区左上角在DEM矩阵中的位置
    yoffset1 = int((originY - pointUL['lat']) / pixelWidth)
    xoffset1 = int((pointUL['lon'] - originX) / (-pixelHight))

    # 研究区右下角在DEM矩阵中的位置
    yoffset2 = int((originY - pointDR['lat']) / pixelWidth)
    xoffset2 = int((pointDR['lon'] - originX) / (-pixelHight))

    # 研究区矩阵行列数
    xx = xoffset2 - xoffset1
    yy = yoffset2 - yoffset1
    # 读取研究区内的数据，并计算高程
    DEMRasterData = DEMBand.ReadAsArray(xoffset1, yoffset1, xx, yy)

    MeanAltitude = np.mean(DEMRasterData)
    return MeanAltitude

class PredefinedWavelength:
    MAX_ALLOWABLE_WAVELENGTH = 4
    MIN_ALLOWABLE_WAVELENGTH = 0.2

    # New predefined wavelengths that I've added to Py6S
    # CONSTANT_NAME = (ID, Start Wavelength, End Wavelength, Filter Function)
    # Note: IDs must be > 1 for new predefined wavelengths

    # Himawari8
    # Interpolated to 2.5nm intervals, as required by 6S
    Himawari_640 = (
        1,
0.555185432,
0.720185432000000,
        np.array(
            [2.61000000e-05, 1.73255547e-05, 1.58768198e-05, 3.25161714e-05,
       4.93541130e-05, 1.05453599e-04, 1.12674113e-04, 1.98055338e-04,
       4.62539240e-04, 9.11719760e-04, 2.16878010e-03, 9.19555465e-03,
       3.43291024e-02, 6.63384806e-02, 1.44663949e-01, 2.23297273e-01,
       3.06302515e-01, 4.26839721e-01, 5.71982358e-01, 7.54295338e-01,
       8.73643886e-01, 8.99100527e-01, 9.06371918e-01, 9.33349043e-01,
       9.46849683e-01, 9.34504761e-01, 9.47755185e-01, 9.70563937e-01,
       9.53331648e-01, 9.36214060e-01, 9.61574287e-01, 9.77890490e-01,
       9.62427788e-01, 9.81450166e-01, 9.98420051e-01, 9.83229272e-01,
       9.85030364e-01, 9.98216881e-01, 9.98789094e-01, 9.93100356e-01,
       9.91816680e-01, 9.94517056e-01, 9.86160301e-01, 9.84987178e-01,
       9.88093977e-01, 9.60604877e-01, 8.89217283e-01, 8.13384175e-01,
       7.15941060e-01, 6.19861039e-01, 5.23217765e-01, 3.84034543e-01,
       2.09031884e-01, 8.41427824e-02, 3.18039946e-02, 1.25337577e-02,
       5.36552073e-03, 2.28879763e-03, 9.78327642e-04, 5.29034786e-04,
       3.13544430e-04, 2.04984489e-04, 1.44426061e-04, 9.66000000e-05,
       7.36775462e-05, 7.09130950e-05, 3.92061466e-05]
        ),
)
    Himawari_860 = (
        2,
0.805217811,
0.912717811,
        np.array(
            [-2.23000000e-05,  1.76277344e-04, -1.23024190e-04,  1.32248508e-04,
        1.24002474e-04,  8.48884472e-05,  4.36389161e-04,  2.20245019e-03,
        5.01912191e-03,  9.12187477e-03,  2.48402819e-02,  9.09182022e-02,
        2.32509392e-01,  4.05330728e-01,  5.76703635e-01,  7.35222279e-01,
        8.69092510e-01,  9.28789513e-01,  9.13958016e-01,  8.72718200e-01,
        8.71065365e-01,  8.88066044e-01,  9.28791391e-01,  9.79790149e-01,
        9.97733339e-01,  8.62180903e-01,  7.09591782e-01,  5.49797065e-01,
        4.05099841e-01,  2.59128316e-01,  1.25608767e-01,  5.20061369e-02,
        2.32136948e-02,  7.31315232e-03,  2.81041176e-03,  4.72544889e-04,
        3.08706842e-04,  8.98819813e-05,  1.73604401e-04,  1.84642246e-04,
        5.38149802e-05,  1.69894797e-04,  1.08178297e-05,  4.09020326e-04]
        ),
    )
    Himawari_1600 = (
        3,
1.545833977,
1.675833977,
        np.array(
            [4.40009000e-04, 4.67372179e-04, 6.89620247e-04, 9.24639485e-04,
       1.01346030e-03, 1.60304207e-03, 2.59425033e-03, 3.99954613e-03,
       6.55986823e-03, 1.12734452e-02, 2.01827708e-02, 3.75369579e-02,
       6.81263259e-02, 1.15209846e-01, 1.82129285e-01, 2.64462308e-01,
       3.55372308e-01, 4.57377140e-01, 5.59941794e-01, 6.73109760e-01,
       7.74188745e-01, 8.46856185e-01, 8.75470541e-01, 8.63914195e-01,
       8.52089503e-01, 8.82743310e-01, 9.39960522e-01, 9.94055858e-01,
       9.81514453e-01, 9.07411511e-01, 8.40205886e-01, 7.56977073e-01,
       6.67030166e-01, 5.74413530e-01, 4.78740066e-01, 3.79476724e-01,
       2.86045363e-01, 2.05021676e-01, 1.39568274e-01, 9.00124363e-02,
       5.74843833e-02, 3.59094117e-02, 2.24146418e-02, 1.38841001e-02,
       9.01341017e-03, 5.71941563e-03, 3.95072608e-03, 2.64594279e-03,
       1.80157439e-03, 1.11443086e-03, 9.78255709e-04, 7.66931481e-04,
       5.23778303e-04]
        ),)
    ## GK2A
    ## Almost Same SRF
    GK2A_640 = (
        4,
        0.55,
0.71,
        np.array(
[1.10000000e-05, 7.26491447e-06, 1.75767560e-05, 1.93124428e-05,
       2.42000000e-05, 4.46842012e-05, 5.30607888e-05, 8.94374412e-05,
       1.32668568e-04, 1.93098858e-04, 4.27405845e-04, 8.76395221e-04,
       2.00828646e-03, 8.28501397e-03, 3.20606202e-02, 6.21480723e-02,
       1.36771125e-01, 2.15219313e-01, 2.98843027e-01, 4.20027643e-01,
       5.66829420e-01, 7.44997247e-01, 8.69246537e-01, 8.97689092e-01,
       9.03964930e-01, 9.28132040e-01, 9.45146945e-01, 9.36092436e-01,
       9.46571421e-01, 9.70767011e-01, 9.59991054e-01, 9.43032584e-01,
       9.62471704e-01, 9.78432177e-01, 9.64480397e-01, 9.82103729e-01,
       9.99144858e-01, 9.77639736e-01, 9.68841570e-01, 9.69028145e-01,
       9.68670593e-01, 9.77936033e-01, 9.91516855e-01, 9.99645381e-01,
       9.85291593e-01, 9.78328627e-01, 9.76483097e-01, 9.45311562e-01,
       8.69211097e-01, 7.88850486e-01, 6.83200232e-01, 5.79509686e-01,
       4.89050620e-01, 3.71255639e-01, 2.10940837e-01, 8.67355295e-02,
       3.24484694e-02, 1.25035572e-02, 5.33732754e-03, 2.29928934e-03,
       1.01078832e-03, 5.19925972e-04, 3.17088073e-04, 2.16771840e-04,
       1.56169780e-04, 1.12625475e-04, 9.73566695e-05, 8.66641526e-05]
        ),
)
    GK2A_856 = (
        5,
0.8,
0.950000000000000,
        np.array(
            [-1.01971000e-04, -7.16802785e-05, -2.87222921e-05, -1.12412516e-04,
        3.37864408e-05, -1.87816653e-05,  1.63322679e-05, -1.75776409e-06,
        2.15933669e-04,  1.26945610e-04,  1.68284786e-04,  1.56249611e-03,
        2.09580291e-03,  5.36611309e-03,  3.47127165e-02,  8.93614796e-02,
        1.80807771e-01,  3.06330226e-01,  4.50901628e-01,  6.05110338e-01,
        7.39848696e-01,  8.30242243e-01,  8.77902794e-01,  9.09298535e-01,
        9.99998510e-01,  9.64133306e-01,  9.51997031e-01,  9.35228915e-01,
        9.16581108e-01,  8.97281168e-01,  8.16224123e-01,  6.88127602e-01,
        5.18839168e-01,  3.23188262e-01,  1.54008020e-01,  5.45305621e-02,
        1.93682409e-02,  9.12994043e-03,  4.98331657e-03,  2.41586310e-03,
        1.22502884e-03,  4.61231644e-04,  2.96543133e-04,  3.93587136e-04,
        4.24439898e-05,  7.22353543e-05,  3.28028630e-04,  1.25743422e-04,
        2.48235871e-04,  7.88601613e-06, -6.93080209e-05, -2.92799056e-05,
        5.08215941e-05, -3.11861916e-05, -4.36042416e-05, -3.74852354e-05,
        1.16001877e-05, -3.96000000e-05,  3.09004974e-05,  1.73381422e-05,
        8.31897541e-05]
        ),
    )
    GK2A_1600 = (
        6,
1.540120129,
1.667620129,
        np.array(
            [3.11400000e-04, 3.22604238e-04, 3.61177031e-04, 5.35708682e-04,
       6.95253092e-04, 9.49834083e-04, 9.87768682e-04, 1.65749296e-03,
       2.63813214e-03, 4.10261156e-03, 6.59021872e-03, 1.12856051e-02,
       2.01452346e-02, 3.71132130e-02, 6.75384926e-02, 1.14102783e-01,
       1.79731420e-01, 2.62298783e-01, 3.52234297e-01, 4.53276258e-01,
       5.55733569e-01, 6.67324342e-01, 7.70668017e-01, 8.50862719e-01,
       8.91954223e-01, 8.82822515e-01, 8.59509349e-01, 8.88211584e-01,
       9.42201420e-01, 9.96944761e-01, 9.84169899e-01, 9.18060444e-01,
       8.49783655e-01, 7.64543331e-01, 6.76071882e-01, 5.81324328e-01,
       4.84424692e-01, 3.82699276e-01, 2.88344099e-01, 2.06028429e-01,
       1.41014649e-01, 9.02575093e-02, 5.76516563e-02, 3.60625086e-02,
       2.23794946e-02, 1.39641025e-02, 8.89070287e-03, 5.86333562e-03,
       3.93536324e-03, 2.66335465e-03, 1.76874972e-03, 1.10682340e-03]
        ),
        
)
# Rayleigh Correction

Rayleigh Correction Based on py6s

Aerosols and Rayleigh reflection account for more than 80% signal in the field of ocean color remote sensing.

Until now, the aerosols correction still could result in huge error especially in the coastal and inland area. Alternatively, the Raleigh corrected reflectance is another choice instead of remote sensing reflectance, since it is very mature.

SeaDAS is the official software of ocean color group to do the atmosphere correction include Rayleigh correction. However, it is not very easy to use and install.

This repo include several commonly used polar-orbit and geostationary Rayleigh correction implementation based on py6s.

Additionally, since the floating algae index is one of the most impotant application of Rayleigh corrected reflectance, I added some additional script to produce floating algae index in the inland lake Taihu, China.

Current Status: Himawari-8, GK2A

In Progress: MODIS-aqua, HY1/COCTS, Sentinel 3/OLCI, GCOM-C/SGLI,GOCI/COMS

Final: Validation with SeaDAS result.

Looking forward to your email and looking for partners.

Current problem to do

1. start from H8 and gk2a raw data

2. the coordination of h8 and gk2a

3. Standardization the code
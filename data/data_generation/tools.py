import os
import gc

import numpy as np
from nbodykit.lab import *
import fitsio
import asdf
import logging

from nbodykit.source.catalog import FITSCatalog
from pmesh.pm import ParticleMesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def CompensateTSC(w, v):
    """
    TSC filter
    """
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 3
        v = v / tmp
    return v

def get_mesh(pos_parts_fns, N_dim, Lbox, interlaced):
    """
    Takes FITS filenames to create meshes
    """
    # create catalog from fitsfile
    cat = FITSCatalog(pos_parts_fns, ext='Data') 
    mesh = cat.to_mesh(window='tsc', Nmesh=N_dim, BoxSize=Lbox, value='Weight', interlaced=interlaced, compensated=False)
    compensation = CompensateTSC # mesh.CompensateTSC not working
    mesh = mesh.apply(compensation, kind='circular', mode='complex')

    return mesh

def get_cross_ps(first_mesh, second_mesh, dk=None):
    """
    Takes two meshes to compute their cross-correlation
    """
    r_cross = FFTPower(first=first_mesh, second=second_mesh, mode='1d', dk=dk)
    Pk_cross = r_cross.power['power']#.real
    ks = r_cross.power['k'] # [Mpc/h]^-1
    P_sn = r_cross.attrs['shotnoise']
    return ks, Pk_cross


def get_Pk(pos1_fns, N_dim, Lbox, interlaced, dk=None, pos2_fns=None):
    """
    Takes FITS filenames of positions to compute the power spectrum
    """
    # calculate power spectrum of the galaxies or halos
    mesh1 = get_mesh(pos1_fns, N_dim, Lbox, interlaced)

    if pos2_fns is None:
        mesh2 = None
    else:
        mesh2 = get_mesh(pos2_fns, N_dim, Lbox, interlaced)
        
    # obtain the "truth"
    r = FFTPower(first=mesh1, second=mesh2, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    return ks, Pk


def get_Pk_splits(pos_gal_s1_fns, pos_gal_s2_fns, pos_mat_fns, N_dim, Lbox, interlaced, dk=None):
    """
    Takes FITS filenames of positions to compute the power spectrum
    """

    # Get meshes for both data splits
    mesh_gal_s1 = get_mesh(pos_gal_s1_fns, N_dim, Lbox, interlaced)
    mesh_gal_s2 = get_mesh(pos_gal_s2_fns, N_dim, Lbox, interlaced)

    logger.info('Created meshes.')

    # Create pm fields that can be added
    field_gal_s1 = mesh_gal_s1.paint(mode='real')
    logger.info('Created field_gal_s1.')
    field_gal_s2 = mesh_gal_s2.paint(mode='real')

    logger.info('Created field_gal_s2.')

    del mesh_gal_s1, mesh_gal_s2
    gc.collect()

    # dictionary with all power spectra
    power_dic = {}

    # obtain all power spectra
    # Auto-power spectrum of HS mesh
    # Create pm fields that can be added
    field_HS = 0.5 * (field_gal_s1 + field_gal_s2)
    logger.info('Created field_HS.')
    # Convert back to meshes
    mesh_HS = FieldMesh(field_HS)
    logger.info('Created mesh_HS.')
    del field_HS
    gc.collect()
    r = FFTPower(first=mesh_HS, second=mesh_HS, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    power_dic['Pk_HSxHS'] = Pk
    power_dic['Pk_HSxHS_SN'] = P_sn
    print("Computed HSxHS power")
    del mesh_HS
    gc.collect()
    # np.save(data_dir+"/Pk_gg.npy", Pk)

    # Auto-power spectrum of HD mesh
    # Create pm fields that can be added
    field_HD = 0.5 * (field_gal_s1 - field_gal_s2)
    logger.info('Created field_HD.')
    # Convert back to meshes
    mesh_HD = FieldMesh(field_HD)
    logger.info('Created mesh_HD.')
    del field_HD
    gc.collect()
    r = FFTPower(first=mesh_HD, second=mesh_HD, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    power_dic['Pk_HDxHD'] = Pk
    power_dic['Pk_HDxHD_SN'] = P_sn
    print("Computed HDxHD power")
    del mesh_HD
    gc.collect()
    # np.save(data_dir+"/Pk_gg.npy", Pk)

    # Cross-power spectrum of s1, s2 mesh
    # Convert back to meshes
    mesh_gal_s1 = FieldMesh(field_gal_s1)
    mesh_gal_s2 = FieldMesh(field_gal_s2)
    logger.info('Created mesh_gal_s1, mesh_gal_s2.')
    del field_gal_s1, field_gal_s2
    gc.collect()
    r = FFTPower(first=mesh_gal_s1, second=mesh_gal_s2, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    power_dic['Pk_s1xs2'] = Pk
    power_dic['Pk_s1xs2_SN'] = P_sn
    print("Computed s1xs2 power")
    # np.save(data_dir+"/Pk_gg.npy", Pk)

    mesh_mat = get_mesh(pos_mat_fns, N_dim, Lbox, interlaced)
    # Cross-power spectrum of s1, m mesh
    r = FFTPower(first=mesh_gal_s1, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    power_dic['Pk_s1xm'] = Pk
    power_dic['Pk_s1xm_SN'] = P_sn
    print("Computed s1xm power")
    del mesh_gal_s1
    gc.collect()
    # np.save(data_dir+"/Pk_gm.npy", Pk)

    # Cross-power spectrum of s1, m mesh
    r = FFTPower(first=mesh_gal_s2, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    power_dic['Pk_s2xm'] = Pk
    power_dic['Pk_s2xm_SN'] = P_sn
    del mesh_mat, mesh_gal_s2
    gc.collect()
    print("Computed s2xm power")
    # np.save(data_dir+"/Pk_gm.npy", Pk)

    return power_dic

def get_all_cf(pos_gal_fns, pos_mat_fns, Lbox):
    """
    Takes FITS filenames of positions to compute the correlation function
    """

    # Define radial bins
    r = np.logspace(-2, 2, 20)

    # calculate correlation function of the galaxies or halos
    # create catalog from fitsfile
    cat_gal = FITSCatalog(pos_gal_fns, ext='Data')

    # dictionary with all power spectra
    cf_dic = {}

    # obtain all power spectra
    cf_obj = SimulationBox2PCF(mode='1d', data1=cat_gal, data2=cat_gal, edges=r, BoxSize=Lbox)
    corr = cf_obj.corr
    D1D2 = cf_obj.D1D2
    # D1R2 = cf_obj.D1R2.astype(np.float64)
    # D2R1 = cf_obj.D2R1.astype(np.float64)
    R1R2 = cf_obj.R1R2
    cf_dic['corr_gg'] = corr['corr']
    cf_dic['D1D2_gg'] = D1D2['npairs']
    # cf_dic['D1R2_gg'] = D1R2
    # cf_dic['D2R1_gg'] = D2R1
    cf_dic['R1R2_gg'] = R1R2['npairs']
    print("Computed gg correlation function.")
    # np.save(data_dir+"/Pk_gg.npy", Pk)
    cf_dic['rs'] = corr['r']
    del cat_gal
    gc.collect()

    # # create catalog from fitsfile
    # cat_mat = FITSCatalog(pos_mat_fns, ext='Data')
    # cf_obj = SimulationBox2PCF(mode='1d', data1=cat_gal, data2=cat_mat, edges=r, BoxSize=Lbox)
    # corr = cf_obj.corr
    # D1D2 = cf_obj.D1D2
    # # D1R2 = cf_obj.D1R2.astype(np.float64)
    # # D2R1 = cf_obj.D2R1.astype(np.float64)
    # R1R2 = cf_obj.R1R2
    # cf_dic['corr_gm'] = corr['corr']
    # cf_dic['D1D2_gm'] = D1D2['npairs']
    # # cf_dic['D1R2_gm'] = D1R2
    # # cf_dic['D2R1_gm'] = D2R1
    # cf_dic['R1R2_gm'] = R1R2['npairs']
    # cf_dic['rs'] = corr['r']
    # del cat_gal, cat_mat
    # gc.collect()
    # print("Computed gm Correlation function.")

    return cf_dic

def get_all_Pk(pos_gal_fns, pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None):
    """
    Takes FITS filenames of positions to compute the power spectrum
    """
    
    # calculate power spectrum of the galaxies or halos
    mesh_gal = get_mesh(pos_gal_fns, N_dim, Lbox, interlaced)

    # dictionary with all power spectra
    power_dic = {}

    # obtain all power spectra
    r = FFTPower(first=mesh_gal, second=mesh_gal, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_dic['Pk_gg'] = Pk
    print("Computed gg power")
    #np.save(data_dir+"/Pk_gg.npy", Pk)

    mesh_mat = get_mesh(pos_mat_fns, N_dim, Lbox, interlaced)
    r = FFTPower(first=mesh_gal, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_dic['Pk_gm'] = Pk
    del mesh_mat
    gc.collect()
    print("Computed gm power")
    #np.save(data_dir+"/Pk_gm.npy", Pk)
    
    mesh_den = load_bigfile(dens_dir, N_dim)
    r = FFTPower(first=mesh_gal, second=mesh_den, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_dic['Pk_gIC'] = Pk
    power_dic['ks'] = ks
    del mesh_den
    gc.collect()
    print("Computed gIC power")
    #np.save(data_dir+"/Pk_gIC.npy", Pk)
        
    return power_dic

def get_mat_Pk(pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None):
    """
    Takes FITS filenames of positions to compute the power spectrum
    """
    # dictionary with all power spectra
    power_dic = {}

    mesh_mat = get_mesh(pos_mat_fns, N_dim, Lbox, interlaced)
    r = FFTPower(first=mesh_mat, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_dic['Pk_mm'] = Pk
    gc.collect()
    print("Computed mm power")
    
    mesh_den = load_bigfile(dens_dir, N_dim)
    r = FFTPower(first=mesh_mat, second=mesh_den, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_dic['Pk_mIC'] = Pk
    power_dic['ks'] = ks
    del mesh_den, mesh_mat
    gc.collect()
    print("Computed mIC power")
        
    return power_dic


def get_all_red_Pk(pos_gal_all_fns, pos_gal_red_fns, pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None):
    """
    Takes FITS filenames of positions to compute the power spectrum
    """
    
    # calculate power spectrum of the galaxies or halos
    mesh_all_gal = get_mesh(pos_gal_all_fns, N_dim, Lbox, interlaced)
    mesh_red_gal = get_mesh(pos_gal_red_fns, N_dim, Lbox, interlaced)
    print("Loaded all galaxies")
    
    # dictionary with all power spectra
    power_all_dic = {}
    power_red_dic = {}
    
    '''
    mesh_mat = get_mesh(pos_mat_fns, N_dim, Lbox, interlaced)
    mesh_den = load_bigfile(dens_dir, N_dim)
    r = FFTPower(first=mesh_den, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_all_dic['Pk_mIC'] = Pk
    np.save("Pk_mIC.npy", Pk)
    print("Computed mIC power")
    quit()
    '''
    
    # obtain all power spectra
    r = FFTPower(first=mesh_all_gal, second=mesh_all_gal, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_all_dic['Pk_gg'] = Pk
    print("Computed gg power 1")
    
    r = FFTPower(first=mesh_red_gal, second=mesh_red_gal, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_red_dic['Pk_gg'] = Pk
    print("Computed gg power 2")
    
    mesh_mat = get_mesh(pos_mat_fns, N_dim, Lbox, interlaced)
    print("Loaded particles")
    
    r = FFTPower(first=mesh_all_gal, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_all_dic['Pk_gm'] = Pk
    print("Computed gm power 1")
    
    r = FFTPower(first=mesh_red_gal, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_red_dic['Pk_gm'] = Pk
    print("Computed gm power 2")

    '''
    r = FFTPower(first=mesh_mat, second=mesh_mat, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_all_dic['Pk_mm'] = Pk
    power_red_dic['Pk_mm'] = Pk
    print("Computed mm power")
    '''
    del mesh_mat
    gc.collect()
    
    mesh_den = load_bigfile(dens_dir, N_dim)
    print("Loaded density field")
    r = FFTPower(first=mesh_all_gal, second=mesh_den, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_all_dic['Pk_gIC'] = Pk
    power_all_dic['ks'] = ks
    print("Computed gIC power 1")

    r = FFTPower(first=mesh_red_gal, second=mesh_den, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    power_red_dic['Pk_gIC'] = Pk
    power_red_dic['ks'] = ks
    print("Computed gIC power 2")

    del mesh_den
    gc.collect()
        
    return power_all_dic, power_red_dic

def get_Pk_arr(pos1, N_dim, Lbox, interlaced, dk=None,pos2=None):
    """
    Takes numpy array(s) with positions to compute the power spectrum.
    I don't think it is parallelizable
    """
    first = {}
    first['Position'] = pos1

    # create mesh object
    cat = ArrayCatalog(first)
    mesh1 = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
    compensation = CompensateTSC # mesh1.CompensateTSC not working
    mesh1 = mesh1.apply(compensation, kind='circular', mode='complex')

    if pos2 is None:
        mesh2 = None
    else:
        second = {}
        second['Position'] = pos2

        # create mesh object
        cat = ArrayCatalog(second)
        mesh2 = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
        compensation = CompensateTSC # mesh2.CompensateTSC not working
        mesh2 = mesh2.apply(compensation, kind='circular', mode='complex')

    # obtain the "truth"
    r = FFTPower(first=mesh1, second=mesh2, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    return ks, Pk

def load_bigfile(dens_dir, N_dim):
    """
    Load bigfile as mesh
    """
    mesh = BigFileMesh(dens_dir+"density_%d.bigfile"%N_dim, mode='real', dataset='Field')
    return mesh

def write_fits(pos, name_pos, data_dir, value=None, mass=None):
    """
    Given a positions array and location, record into fits file
    """
    dtype = [('Position', ('f8', 3))]
    if value is not None:
        dtype.append(('Value', 'f8'))
    if mass is not None:
        dtype.append(('Mass', 'f8'))
    data = np.empty(pos.shape[0], dtype=dtype)
    
    if value is not None:    
        data['Value'] = value
    if mass is not None:
        data['Mass'] = mass
    data['Position'] = pos
    
    # write to a FITS file using fitsio
    fitsio.write(data_dir+"pos_"+name_pos+".fits", data, extname='Data')
    return

def write_asdf(data_dict, filename, save_dir, header=None):
    # create data tree structure                                                                                                                                                                                                                       
    data_tree = {}
    data_tree['data'] = data_dict
    if header != None:
        data_tree['header'] = header

    # save the data and close file
    output_file = asdf.AsdfFile(data_tree)
    output_file.write_to(os.path.join(save_dir,filename))
    output_file.close()

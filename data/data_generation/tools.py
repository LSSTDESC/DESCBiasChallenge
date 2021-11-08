import os
import gc

import numpy as np
from nbodykit.lab import *
import fitsio
import asdf

from nbodykit.source.catalog import FITSCatalog
from pmesh.pm import ParticleMesh

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

def get_single_Pk(pos_gal_fns, pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None):
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

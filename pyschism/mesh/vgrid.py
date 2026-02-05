from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.pyplot import * # neeed to merge this back to plt

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union
from tqdm import tqdm
from functools import lru_cache
import pathlib, os
import subprocess
import tempfile
import warnings
import numpy as np
from pyschism.mesh.hgrid import Hgrid

def C_of_sigma(sigma, theta_b, theta_f):
    assert theta_b <= 0. and theta_b <= 1.
    assert theta_f <= 0. and theta_f <= 1.
    A = (1-theta_b)(np.sinh(sigma*theta_f)/np.sinh(theta_f))
    B_1 = np.tanh(theta_f*(sigma+0.5)) - np.tanh(theta_f/2.)
    B = theta_b * (B_1 / (2.*np.tanh(theta_f/2.)))
    return A + B


def eta_of_sigma(sigma):
    return 1 + sigma


def S_to_Z(sigma):
    # eq 3.1
    pass


class VgridType(Enum):

    LSC2 = 1
    SZ = 2

    @classmethod
    def _missing_(cls, name):
        raise ValueError(f'ivcor={name} is not a valid vgrid type.')


class Vgrid(ABC):

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    
    @staticmethod
    def default():
        return SZ.default() 

    @staticmethod
    def v2d(h_s, ztot, h_c, theta_b, theta_f, sigma):
        return SZ._v2d(h_s, ztot, h_c, theta_b, theta_f, sigma)

    @classmethod
    def from_binary(cls, hgrid, binary='gen_vqs'):
        _tmpdir = tempfile.TemporaryDirectory()
        tmpdir = pathlib.Path(_tmpdir.name)
        hgrid = Hgrid.open(hgrid, crs='EPSG:4326')
        hgrid.write(tmpdir / 'hgrid.gr3')
        subprocess.check_call([binary], cwd=tmpdir)
        return cls.open(tmpdir / 'vgrid.in')

    @staticmethod
    def open(path):
        '''
        Based on:
        https://github.com/wzhengui/pylibs/blob/master/Utility/schism_file.py
        '''
        with open(path) as f:
            return VgridTypeDispatch[VgridType(
                int(f.read().strip().split()[0])).name].value.open(path)

    @abstractmethod
    def get_xyz(self, gr3, crs=None):
        pass

    @property
    def ivcor(self):
        return VgridType[self.__class__.__name__].value

    @property
    @abstractmethod
    def nvrt(self):
        raise NotImplementedError

    @lru_cache(maxsize=1)
    def is2D(self):
        if isinstance(self, SZ):
            if str(self) == str(SZ.default()):
                return True
        return False

    def is3D(self):
        return not self.is2D()


class LSC2_original(Vgrid):

    def __init__(self, hsm, nv, h_c, theta_b, theta_f, sigma=None):
        '''
        Todo: Consider intializing from sigma only.
        The other parameters can go to a factory method.
        '''
        self.hsm = np.array(hsm)
        self.nv = np.array(nv)
        self.h_c = h_c
        self.theta_b = theta_b
        self.theta_f = theta_f
        self.m_grid = None
        self._znd = None
        self._nlayer = None
        self._snd = None
        self.sigma = sigma  # expose sigma for backward compatibility

    @classmethod
    def from_sigma(cls, sigma):
        '''
        Initialize the LSC2 class using the sigma values
        mainly for 'def open' method.
        Consider setting this as __init__ and remove other parameters.

        sigma: np.ndarray of shape (n, m), where
            n: number of horizontal nodes
            m: number of vertical layers
        '''
        return cls(hsm=None, nv=None, h_c=None, theta_b=None, theta_f=None, sigma=sigma)

    def __str__(self):
        f = [
            f'{self.ivcor}',
            f'{self.nvrt}',
        ]
        for i, row in enumerate(self.sigma):
            kbp = int((row == -1).sum())
            line = [
                f'{i+1}'.rjust(11),
                f'{kbp}'.rjust(11),
                7*' ',
                '-1.000000',
            ]
            for value in row:
                if value != -1:
                    line.append(7*' ')
                    line.append(f'{value:6f}')

            f.append(' '.join(line))
        return '\n'.join(f)

    def get_xyz(self, gr3, crs=None):
        if type(gr3) == Hgrid:
            gr3 = gr3
        else:
            gr3=Hgrid.open(gr3)
        xy = gr3.get_xy(crs)
        z = gr3.values[:, None]*self.sigma
        x = np.tile(xy[:, 0], (z.shape[1],))
        y = np.tile(xy[:, 0], (z.shape[1],))
        return np.vstack([x, y, z.flatten()]).T

    def calc_m_grid(self):
        '''
        create master grid
        Adapted from:
        https://github.com/wzhengui/pylibs/blob/master/pyScripts/gen_vqs.py
        '''
        if self.m_grid:
            pass
        else:
            z_mas=np.ones([self.nhm,self.nv[-1]])*np.nan; eta=0.0
            for m, [hsmi,nvi] in enumerate(zip(self.hsm,self.nv)):
                #strethcing funciton
                hc=min(hsmi,self.h_c)
                for k in np.arange(nvi):
                    sigma= k/(1-nvi)  #zi=-sigma #original sigma coordinate
                    #compute zcoordinate
                    cs=(1-self.theta_b)*np.sinh(self.theta_f*sigma)/np.sinh(self.theta_f)+\
                        self.theta_b*(np.tanh(self.theta_f*(sigma+0.5))-\
                                np.tanh(self.theta_f*0.5))/2/np.tanh(self.theta_f*0.5)
                    z_mas[m,k]=eta*(1+sigma)+hc*sigma+(hsmi-hc)*cs

                #normalize z_mas
                z_mas[m]=-(z_mas[m]-z_mas[m,0])*hsmi/(z_mas[m,nvi-1]-z_mas[m,0])
                if min(z_mas[m,:self.nv[m]]-z_mas[m+1,:self.nv[m]])<0: 
                    warnings.warn('check: master grid layer={}, hsm={}, nv={}'.format(m+1,self.hsm[m+1],self.nv[m+1]))
            self.m_grid = z_mas
            # self.z_mas = z_mas
            # self.s_mas=np.array([z_mas[i]/self.hsm[i] for i in np.arange(self.nhm)])
    
    def make_m_plot(self, axes=None):
        '''
        plot master grid
        Adapted from:
        https://github.com/wzhengui/pylibs/blob/master/pyScripts/gen_vqs.py
        '''   
        import matplotlib.pyplot as plt
        for i in np.arange(self.nhm - 1):
            if np.min(self.m_grid[i, :self.nv[i]] - self.m_grid[i + 1, :self.nv[i]]) < 0:
                print(f'check: master grid layer={i+1}, hsm={self.hsm[i+1]}, nv={self.nv[i+1]}')
        if axes is None:
            fig, axes = plt.subplots(figsize=[10, 5])
        for i in np.arange(self.nhm): 
            axes.plot(i * np.ones(self.nv[i]), self.m_grid[i, :self.nv[i]], 'k-', lw=0.3)
        for k in np.arange(self.nv[-1]): 
            axes.plot(np.arange(self.nhm), self.m_grid.T[k], 'k-', lw=0.3)
        axes.set_xlim([-0.5, self.nhm - 0.5])
        axes.set_ylim([-self.hsm[-1], 0.5])
        axes.set_title("Master VGrid")
        axes.set_xlabel("Vertical Column Index")
        axes.set_ylabel("Depth [m]")

    def calc_lsc2_att(self, gr3, dz_bot_min=0.1, dt_crs=None):
        '''
        master grid to lsc2:
        compute vertical layers at nodes
        gr3 is either file or Hgrid Object
        Adapted from:
        https://github.com/wzhengui/pylibs/blob/master/pyScripts/gen_vqs.py
        '''
        if type(gr3) == Hgrid:
            gd = gr3
        else:
            gd=Hgrid.open(gr3)
        dp = gd.values*-1
        fpz=dp<self.hsm[0]
        dp[fpz]=self.hsm[0]
        
        #find hsm index for all points
        rat=np.ones(len(gd.nodes.id))*np.nan
        nlayer=np.zeros(len(gd.nodes.id)).astype('int')
        ind1=np.zeros(len(gd.nodes.id)).astype('int')
        ind2=np.zeros(len(gd.nodes.id)).astype('int')
        for m, hsmi in enumerate(self.hsm):
            if m==0:
                fp=dp<=self.hsm[m]
                ind1[fp]=0; 
                ind2[fp]=0
                rat[fp]=0; 
                nlayer[fp]=self.nv[0]
            else:
                fp=(dp>self.hsm[m-1])*(dp<=self.hsm[m])
                ind1[fp]=m-1
                ind2[fp]=m
                rat[fp]=(dp[fp]-self.hsm[m-1])/(self.hsm[m]-self.hsm[m-1]) # get ratio of neighboring master grids
                nlayer[fp]=self.nv[m]

        #Find the last non NaN node and fill with NaN values
        last_non_nan = (~np.isnan(self.m_grid)).cumsum(1).argmax(1)

        z_mas=np.array([np.nan_to_num(z_mas_arr,nan=z_mas_arr[last_non_nan[i]]) for i, z_mas_arr in enumerate(self.m_grid)])
        # znd=z_mas[ind1]*(1-rat[:,None])+z_mas[ind2]*rat[:,None]; #z coordinate using ratio of neighboring grids
        znd=z_mas[ind2] # z coordinate

        for i in np.arange(len(gd.nodes.id)):
            znd[i,nlayer[i]-1]=-dp[i]
            znd[i,nlayer[i]:]=np.nan
         
        snd=znd/dp[:,None]; #sigma coordinate

        # QC check on vgird

        # check: z must be strictly decreasing with k
        bad_z = (znd[:, :-1] <= znd[:, 1:])          # shape (n_nodes, n_levels-1)
        if np.any(bad_z):
            i, k = np.argwhere(bad_z)[0]             # first offending pair
            raise ValueError(f"wrong vertical layers at node {i}, levels {k}->{k+1}: znd={znd[i,k]} <= {znd[i,k+1]}")

        # check: snd neighbor is >= within 1e-6 (or increasing) 
        bad_sigma_pair = (snd[:, :-1] < snd[:, 1:]) | np.isclose(snd[:, :-1], snd[:, 1:], atol=1e-6, rtol=0.0)

        # NaN the *upper* neighbor (k+1) wherever the pair (k,k+1) is bad:
        bad_sigma = np.zeros_like(snd, dtype=bool)
        bad_sigma[:, 1:] = bad_sigma_pair

        n_bad = int(bad_sigma.sum())
        if n_bad:
            raise ValueError(f"sigma layers too close (within 1e-6)")
                    
        # set properties
        self._znd = znd
        self._snd = snd
        self.sigma = np.fliplr(snd)
        self._nlayer = nlayer

    def write(self, path: os.PathLike, overwrite: bool = False, method: 1 | 2 = 2):
        '''
        Write vgrid.in

        path : output filepath
        overwrite : bool, do / do not overwrite file specified in path if it exists
        method : int, 1 or 2 (1 was original method, 2 is a speed up; intial checks show they give the same file)

        write mg2lsc2 into vgrid.in
        Todo: enable writing from sigma only, to be done with the refactoring of the class.
        '''
        path = pathlib.Path(path)
        if path.is_file() and not overwrite:
            print(f'{path} exists, pass overwrite=True to allow overwrite ... skipping ...')
            return
        
        print('writing:', str(path))

        if method == 1:
            with open(path, 'w') as fid:
                fid.write('           1 !average # of layers={:0.2f}\n          {} !nvrt\n'.format\
                        (np.mean(self._nlayer),self.nvrt))
                bli=[]#bottom level index
                for i in np.arange(len(self._nlayer)):
                    nlayeri=self._nlayer[i]
                    si=np.flipud(self._snd[i,:nlayeri])
                    bli.append(self.nvrt-nlayeri+1)
                    fstr=f"         {self.nvrt-nlayeri+1:2}"
                    fid.write(fstr)
                for i in range(self.nvrt):
                    fid.write(f'\n         {i+1}')
                    for n,bl in enumerate(bli):
                        si=np.flipud(self._snd[n])
                        if bl <= i+1:
                            fid.write(f"      {si[i]:.6f}")
                        else:
                            fid.write(f"      {-9.:.6f}")
                fid.close()

        elif method == 2:

            chunk=200_000
            nvrt = int(self.nvrt)
            n_nodes = int(len(self._nlayer))

            nlayer = np.asarray(self._nlayer, dtype=np.int32)
            bli = (nvrt - nlayer + 1).astype(np.int32)  # 1-based bottom level index
            snd = self._snd                               # (n_nodes, nvrt)

            with open(path, "w", buffering=64 * 1024 * 1024) as fid:
                fid.write(
                    f"           1 !average # of layers={float(nlayer.mean()):0.2f}\n"
                    f"          {nvrt} !nvrt\n"
                )

                # ---- bottom indices line: TEXT mode (sep must be non-empty) + leading spaces
                for s in range(0, n_nodes, chunk):
                    e = min(s + chunk, n_nodes)
                    fid.write("         ")                               # leading spaces before first value in this chunk
                    bli[s:e].tofile(fid, sep="         ", format="%2d")  # 9 spaces between values

                # ---- data lines
                i=0
                for i in tqdm(range(nvrt),leave=False,mininterval=0.2):
                    level = i + 1
                    fid.write(f"\n         {level}")
                    col = nvrt - 1 - i  # equivalent to flipud per-node without copying

                    for s in range(0, n_nodes, chunk):
                        e = min(s + chunk, n_nodes)
                        mask = (bli[s:e] <= level)
                        vals = np.where(mask, snd[s:e, col], -9.0)
                        fid.write("      ")                              # leading spaces before first value in this chunk
                        vals.tofile(fid, sep="      ", format="%.6f")    # 6 spaces between values


    @classmethod
    def open(cls, path):

        path = pathlib.Path(path)

        with open(path) as f:
            lines = f.readlines()

        ivcor = int(lines[0].strip().split()[0])
        if ivcor != 1:
            raise TypeError(f'File {path} is not an LSC2 grid (ivcor != 1).')

        nvrt = int(lines[1].strip().split()[0])

        sline = np.array(lines[2].split()).astype('float')        
        if sline.min() < 0:
            # old version
            kbp = np.array([int(i.split()[1])-1 for i in lines[2:]])
            sigma = -np.ones((len(kbp), nvrt))

            for i, line in enumerate(lines[2:]):
                sigma[i, kbp[i]:] = np.array(
                    line.strip().split()[2:]).astype('float')

        else:
            # new version
            sline = sline.astype('int')
            kbp = sline-1 # index to first sigma coord
            sigma = np.array([line.split()[1:] for line in lines[3:]]).T.astype('float')
            # replace -9. with -1.
            fpm = sigma < -1
            sigma[fpm] = np.nan
            nlayer = nvrt - kbp # number of sigma layers

        obj = cls.from_sigma(sigma)
        obj._nlayer = nlayer
        return obj

    @property
    def nvrt(self):
        return max(self._nlayer)  # may not be equal to self.nv[-1]

    @property
    def nhm(self):
        return self.hsm.shape[0]


class SZ(Vgrid):

    def __init__(self, h_s, ztot, h_c, theta_b, theta_f, sigma):
        self.h_s = h_s
        self.ztot = np.array(ztot)
        self.h_c = h_c
        self.theta_b = theta_b
        self.theta_f = theta_f
        self.sigma = np.array(sigma)

    def __str__(self):
        f = [
            f'{self.ivcor:d} !ivcor',
            f'{self.nvrt:d} {self.kz:d} {self.h_s:G} '
            '!nvrt, kz (# of Z-levels); h_s '
            ' (transition depth between S and Z)',
            'Z levels',
        ]
        for i, row in enumerate(self.ztot):
            f.append(f'{i+1:d} {row:G}')

        f.extend([
            'S levels',
            f'{self.h_c:G} {self.theta_b:G} {self.theta_f:G} '
            ' !h_c, theta_b, theta_f',
            ])
        for i, row in enumerate(self.sigma):
            f.append(f'{i+1:d} {row:G}')
        return '\n'.join(f)

    def get_xyz(self, gr3, crs=None):
        raise NotImplementedError('SZ.get_xyz')

    def write(self, path, overwrite=False):
        path = pathlib.Path(path)
        if path.is_file() and not overwrite:
            print(f'{path} exists, pass overwrite=True to allow overwrite ... skipping ...')
            return
        
        with open(path, 'w') as f:
            f.write(str(self))

    @classmethod
    def open(cls, path):

        path = pathlib.Path(path)

        with open(path) as f:
            lines = f.readlines()

        ivcor = int(lines[0].strip().split()[0])
        if ivcor != 2:
            raise TypeError(f'File {path} is not an SZ grid (ivcor != 2).')

        nvrt = int(lines[1].strip().split()[0])

        kz, h_s = lines[1].strip().split()[1:3]
        kz = int(kz)
        h_s = float(h_s)

        # read z grid
        ztot = []
        irec = 2
        for i in np.arange(kz):
            irec = irec+1
            ztot.append(lines[irec].strip().split()[1])
        ztot = np.array(ztot).astype('float')
        # read s grid
        sigma = []
        irec = irec+2
        nsigma = nvrt - kz+1
        h_c, theta_b, theta_f = np.array(
            lines[irec].strip().split()[:3]).astype('float')
        for i in np.arange(nsigma):
            irec = irec + 1
            sigma.append(lines[irec].strip().split()[1])
        sigma = np.array(sigma).astype('float')
        return cls(h_s, ztot, h_c, theta_b, theta_f, sigma)

    @classmethod
    def default(cls):
        # h_s, ztot, h_c, theta_b, theta_f, sigma
        return cls(1.e6, [-1.e6], 40., 1., 1.e-4, [-1, 0.])
        #return cls(h_s, ztot, h_c, theta_b, theta_f, sigma)

    @classmethod
    def _v2d(cls, h_s, ztot, h_c, theta_b, theta_f, sigma):
        return cls(h_s, ztot, h_c, theta_b, theta_f, sigma)

    @property
    def kz(self):
        return self.ztot.shape[0]

    @property
    def nvrt(self):
        return self.sigma.shape[0]

class LSC2(Vgrid):
    """
    Python translation of the core algorithm in SCHISM Utility/Pre-Processing/gen_vqs_1.f90
    for ivcor=1 (VQS/LSC^2 master grids).

    Conventions (matching the Fortran script):
      - Vertical index k=1 is surface, k=nv is bottom.
      - z is in meters, 0 at MSL-ish, negative below free surface.
      - depth (dp) is positive downward.


    WARNING: not integrated yet into pyschism ...  need to replace LSC2
    """

    def __init__(
        self,
        hsm: np.ndarray,
        nv_vqs: np.ndarray,
        theta_b: Optional[np.ndarray] = None,
        theta_f: Optional[np.ndarray] = None,
        hc: Optional[np.ndarray] = None, # 10.5 is default
        a_vqs0: float = -0.3,
        dz_bot_min: float = 1.0,
        etal: float = 0.0,
        method: str ='roms_vtransform2_vstretching4', # gen_vqs_S or roms_vtransform2_vstretching4
        sigma_vqs: np.ndarray = None,
    ):
        """
        LCS2 init (your requested "LCS2" method == initializer).

        Parameters
        ----------
        hsm : (m,) array
            Master-grid depths (monotonically increasing; positive down).
        nv_vqs : (m,) array
            Number of vertical levels for each master grid.
        theta_b, theta_f : (m,) arrays
            S-stretching parameters used in the master-grid generation (Option 2 in Fortran).
            If None, defaults to theta_b=0 and theta_f=3 (constant).
        a_vqs0 : float
            Quadratic sigma skew used ONLY for shallow region (dp <= hsm[0]) in the Fortran code.
        dz_bot_min : float
            Minimum bottom layer thickness for truncating deeper grids.
        etal : float
            Constant elevation used in the Fortran code (eta2=etal everywhere).
        method : str
            'gen_vqs_S' or 'roms_vtransform2_vstretching4'
        """
        self.hsm = np.asarray(hsm, dtype=float)
        self.nv_vqs = np.asarray(nv_vqs, dtype=int)

        if self.hsm.ndim != 1 or self.nv_vqs.ndim != 1:
            raise ValueError("hsm and nv_vqs must be 1D arrays.")
        if self.hsm.size != self.nv_vqs.size:
            raise ValueError("hsm and nv_vqs must have the same length.")
        if np.any(np.diff(self.hsm) <= 0):
            raise ValueError("hsm must be strictly increasing.")
        if np.any(self.nv_vqs < 2):
            raise ValueError("Each nv_vqs must be >= 2.")

        m = self.hsm.size
        if theta_b is None:
            theta_b = np.zeros(m, dtype=float)
        if theta_f is None:
            theta_f = np.full(m, 3.0, dtype=float)
        if hc is None:
            hc = np.full(m, 10.5, dtype=float)

        self.theta_b = np.asarray(theta_b, dtype=float)
        self.theta_f = np.asarray(theta_f, dtype=float)
        self.hc = np.asarray(hc, dtype=float)

        if self.theta_b.shape != (m,) or self.theta_f.shape != (m,) or self.hc.shape != (m,):
            raise ValueError("theta_b/theta_f must be shape (m_vqs,).")

        self.a_vqs0 = float(a_vqs0)
        self.dz_bot_min = float(dz_bot_min)
        self.etal = float(etal)
        self.method = method
        
        # Outputs (filled after calls)
        self.z_mas: Optional[np.ndarray] = None     # (nvrt_m, m_vqs) with NaN for unused
        self.z_cor: Optional[np.ndarray] = None     # (nvrt_m, np)
        self.kbp: Optional[np.ndarray] = None       # (np,) number of valid levels at each node
        self.m0: Optional[np.ndarray] = None        # (np,) selected master-grid index (0-based)
        self.sigma_vqs: Optional[np.ndarray] = sigma_vqs # (nvrt_m, np)

    @property # to merge into pyschism ... 
    def sigma(self):
        return self.sigma_vqs.T
    
    @property
    def nvrt(self):
        return np.max(self.nv_vqs)  # may not be equal to self.nv[-1]

    # @property
    # def nhm(self):
    #     return self.hsm.shape[0]

    # -------
    # Abstract method implementation: __str__, get_xyz, nvrt
    # ------

    
    def __str__(self):
        raise(NotImplementedError)
    
    def nvrt(self):
        raise NotImplementedError

    def get_xyz(self, gr3, crs=None):
        # if type(gr3) == Hgrid:
        #     gr3 = gr3
        # else:
        #     gr3=Hgrid.open(gr3)
        # xy = gr3.get_xy(crs)
        # z = gr3.values[:, None]*self.sigma
        # x = np.tile(xy[:, 0], (z.shape[1],))
        # y = np.tile(xy[:, 0], (z.shape[1],))
        # return np.vstack([x, y, z.flatten()]).T
        raise(NotImplementedError)

    # -------------------------
    # Master grid
    # -------------------------
    def calc_m_grid(self) -> np.ndarray:
        """
        Compute master-grid z levels z_mas for each hsm, following Option 2 ("S") in the Fortran script.
        The first master grid ends up being linear-sigma because (hsm(m)-hsm(1)) = 0 when m=1.

        Returns
        -------
        z_mas : (nvrt_m, m_vqs) array
            z values (negative below) with NaN for rows beyond nv_vqs[m].
        """
        m_vqs = self.hsm.size
        nvrt_m = int(self.nv_vqs[-1])

        z_mas = np.full((nvrt_m, m_vqs), np.nan, dtype=float)
        # hc = self.hsm[0]
        etal = self.etal

        for m in range(m_vqs):
            if self.method == 'gen_vqs_S':
                z = self.gen_vqs_S(hsm=self.hsm[m], N=self.nv_vqs[m], hc=self.hc[m], theta_f=self.theta_f[m], theta_b=self.theta_b[m])
            elif self.method == 'roms_vtransform2_vstretching4':
                z, s, Cs = self.roms_vtransform2_vstretching4(hsm=self.hsm[m], N=self.nv_vqs[m], hc=self.hc[m], theta_f=self.theta_f[m], theta_b=self.theta_b[m])
            z_mas[:self.nv_vqs[m], m] = z

        self.z_mas = z_mas
        return z_mas

    # -------------------------
    # Node-based zcor
    # -------------------------
    def calc_z_cor(self, hgrid: Hgrid) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vertical coordinates z_cor (znd in Fortran) for each node from master grids.

        Parameters
        ----------
        hgrid : Hgrid
            Horizontal grid (x,y,depth, connectivity). Only x,y,depth are needed for z_cor;
            connectivity is used by plot_nv if you choose.

        Returns
        -------
        z_cor : (nvrt_m, np) array
        kbp   : (np,) int array
            Number of valid levels at each node (bottom index in Fortran).
        """
        if self.z_mas is None:
            self.calc_m_grid()

        assert self.z_mas is not None
        z_mas = self.z_mas
        nvrt_m, m_vqs = z_mas.shape
        hsm = self.hsm
        nv_vqs = self.nv_vqs
        dp = np.asarray(-hgrid.values, dtype=float)
        npnt = dp.size

        # if dp.max() > hsm[-1] + 1e-12:
        #     raise ValueError(f"Max depth ({dp.max():.3f}) exceeds deepest master depth ({hsm[-1]:.3f}).")

        etal = self.etal
        dz_bot_min = self.dz_bot_min

        znd = np.full((nvrt_m, npnt), np.nan, dtype=float)
        sigma_vqs = np.full((nvrt_m, npnt), np.nan, dtype=float)
        nlayer = np.full((npnt,), np.nan, dtype=float)
        kbp = np.zeros(npnt, dtype=int)
        m0 = np.full(npnt, -1, dtype=int)

        # Shallow: dp <= hsm[0]
        shallow = dp <= hsm[0]
        if np.any(shallow):
            nv0 = int(nv_vqs[0])
            k_idx = np.arange(nv0, dtype=float)
            sigma = (k_idx) / (1.0 - nv0)
            sig_t = self.a_vqs0 * sigma * sigma + (1.0 + self.a_vqs0) * sigma  # quadratic transform
            sigma_vqs[:nv0, shallow] = sig_t[:, None]
            znd[:nv0, shallow] = sig_t[:, None] * (etal + dp[shallow])[None, :] + etal
            kbp[shallow] = nv0
            m0[shallow] = 0
            nlayer[shallow]=self.nv_vqs[0]     

        # not shallow : dp > hsm[0]
        deep_idx = np.where(~shallow)[0]
        for i in deep_idx:
            di = dp[i]

        # for i, di in enumerate(dp):
            # find m such that hsm[m-1] < di <= hsm[m], with m in 1..m_vqs-1 (0-based)
            mm = None
            for m in np.arange(m_vqs):
                nlayer[i]=self.nv_vqs[m]     
                if m == 0:
                    mm=m
                    zrat=0; 
                elif di > hsm[m - 1] and di <= hsm[m]:
                    mm=m
                    zrat = (di - hsm[m - 1]) / (hsm[m] - hsm[m - 1])  # (0,1]
                    break
            if mm is None:
                raise RuntimeError(f"Failed to find master grid for node {i}, depth={di}")

            m0[i] = mm
            nv_cur = int(nv_vqs[mm])
            nv_prev = int(nv_vqs[mm - 1])

            bottom_k = None
            for k in range(nv_cur):  # 0-based -> Fortran k=k+1
                z1 = z_mas[min(k, nv_prev - 1), mm - 1]
                z2 = z_mas[k, mm]
                z3 = z1 + (z2 - z1) * zrat

                # keep level if above the truncation threshold
                if z3 >= -di + dz_bot_min:
                    znd[k, i] = z3
                else:
                    bottom_k = k  # this k is where Fortran sets kbp=k and exits
                    break

            if bottom_k is None:
                raise RuntimeError(
                    f"Failed to find a bottom for node {i}, depth={di}. "
                    f"Try increasing dz_bot_min or revisiting master grids."
                )

            # Fortran sets znd(kbp)=-dp, with kbp being 1-based index.
            # Our bottom_k is 0-based index at which we set bottom.
            znd[bottom_k, i] = -di
            kbp[i] = bottom_k + 1  # number of valid levels

            # Order check (strictly decreasing z with k)
            zcol = znd[:kbp[i], i]
            if np.any(zcol[:-1] <= zcol[1:]):
                raise RuntimeError(f"Inverted z at node {i} (depth={di}).")

            # Build sigma_vqs for deep (matching later Fortran output logic)
            sigma_vqs[0, i] = 0.0
            sigma_vqs[kbp[i] - 1, i] = -1.0
            if kbp[i] > 2:
                mid = slice(1, kbp[i] - 1)
                sigma_vqs[mid, i] = (znd[mid, i] - etal) / (etal + di)

            # sigma order check (must be decreasing with k and with tol of 1e-6)
            scol = sigma_vqs[:kbp[i], i]
            if np.any(scol[1:] >= scol[:-1]):
                raise RuntimeError(f"Inverted sigma at node {i} (depth={di}).")

            sigma_vqs_ = sigma_vqs[:kbp[i], i]
            too_close = np.isclose(sigma_vqs_[1:], sigma_vqs_[:-1], atol=1e-6, rtol=0.0)
            if np.any(too_close):
                raise RuntimeError(f"sigma at node {i} (depth={di}) has too close of spacing")

        # # Extend beyond bottom for plotting (like Fortran)
        # for i in range(npnt):
        #     if kbp[i] < nvrt_m:
        #         znd[kbp[i]:, i] = -dp[i]

        self.z_cor = znd
        self.kbp = kbp
        self.m0 = m0
        self.sigma_vqs = sigma_vqs
        self.nlayer = nlayer
        return znd, kbp
    
    @classmethod
    def open(cls, path):

        """
        From pyschism.mesh.vgrid.LSC2
        """

        path = pathlib.Path(path)

        with open(path) as f:
            lines = f.readlines()

        ivcor = int(lines[0].strip().split()[0])
        if ivcor != 1:
            raise TypeError(f'File {path} is not an LSC2 grid (ivcor != 1).')

        nvrt = int(lines[1].strip().split()[0])

        sline = np.array(lines[2].split()).astype('float')        
        if sline.min() < 0:
            # old version
            kbp = np.array([int(i.split()[1])-1 for i in lines[2:]])
            sigma = -np.ones((len(kbp), nvrt))+np.nan

            for i, line in enumerate(lines[2:]):
                sigma[i, kbp[i]:] = np.array(line.strip().split()[2:]).astype('float')
        else:
            # new version
            sline = sline.astype('int')
            kbp = sline-1 # index to first sigma coord
            sigma = np.array([line.split()[1:] for line in lines[3:]]).T.astype('float')
            # replace -9. with -1.
            fpm = sigma < -1
            sigma[fpm] = np.nan
            nlayer = nvrt - kbp # number of sigma layers

        obj = cls.from_sigma(sigma)
        obj._nlayer = nlayer
        return obj
    
    @classmethod
    def from_sigma(cls, sigma_vqs):
        '''
        Initialize the LSC2 class using the sigma values
        mainly for 'def open' method.
        Consider setting this as __init__ and remove other parameters.

        sigma: np.ndarray of shape (n, m), where
            n: number of horizontal nodes
            m: number of vertical layers
        '''
        return cls(hsm=None, nv_vqs=None, h_c=None, theta_b=None, theta_f=None, sigma_vqs=sigma_vqs)
    

    def write(self, path: os.PathLike, overwrite: bool = False):
        '''
        Write vgrid.in

        path : output filepath
        overwrite : bool, do / do not overwrite file specified in path if it exists
        '''
        from tqdm import tqdm 
        path = pathlib.Path(path)
        if path.is_file() and not overwrite:
            print(f'{path} exists, pass overwrite=True to allow overwrite ... skipping ...')
            return

        print('writing:', str(path))

        chunk=200_000
        nvrt = int(self.sigma_vqs.shape[0])
        n_nodes = int(self.sigma_vqs.shape[1])

        nlayer = np.asarray(self.nlayer, dtype=np.int32)
        bli = (nvrt - nlayer + 1).astype(np.int32)  # 1-based bottom level index
        snd = self.sigma_vqs.T                        # (n_nodes, nvrt)

        with open(path, "w", buffering=64 * 1024 * 1024) as fid:
            fid.write(
                f"         1 !average # of layers={float(nlayer.mean()):0.2f}\n"
                f"         {nvrt} !nvrt\n"
            )

            # ---- bottom indices line: TEXT mode (sep must be non-empty) + leading spaces
            for s in range(0, n_nodes, chunk):
                e = min(s + chunk, n_nodes)
                fid.write("         ")                               # leading spaces before first value in this chunk
                bli[s:e].tofile(fid, sep="         ", format="%2d")  # 9 spaces between values

            # ---- data lines
            i=0
            for i in tqdm(range(nvrt),leave=False,mininterval=0.2):
                level = i + 1
                fid.write(f"\n         {level}")
                col = nvrt - 1 - i  # equivalent to flipud per-node without copying

                for s in range(0, n_nodes, chunk):
                    e = min(s + chunk, n_nodes)
                    # mask = (bli[s:e] <= level)
                    mask = ~np.isnan(snd[s:e, col])
                    vals = np.where(mask, snd[s:e, col], -9.0)
                    fid.write("      ")                              # leading spaces before first value in this chunk
                    vals.tofile(fid, sep="      ", format="%.6f")    # 6 spaces between values


    # -------------------------
    # Plotting helpers
    # -------------------------
    def plot_m_grid(self) -> None:
        """
        Plot z_mas vs level index for each master depth.
        """
        if self.z_mas is None:
            self.calc_m_grid()
        assert self.z_mas is not None

        m_vqs = self.hsm.size
        plt.figure()
        for m in range(m_vqs):
            nv = int(self.nv_vqs[m])
            z = self.z_mas[:nv, m]
            plt.plot(np.full_like(z,m), z, marker=".", linewidth=1)
        plt.xlabel("Vertical level k (1=surface)")
        plt.ylabel("z (m)")
        plt.title("Master grids z_mas")
        plt.grid(True, alpha=0.3)
        plt.show()

    def make_m_plot(self, ax=None,annotate=True):
        '''
        plot master grid
        from:
        https://github.com/wzhengui/pylibs/blob/master/pyScripts/gen_vqs.py
        '''   
        import matplotlib.pyplot as plt
        if self.z_mas is None:
            self.calc_m_grid()
        assert self.z_mas is not None
        nhm = self.hsm.shape[0]
        # for i in np.arange(nhm - 1):
        #     if np.min(self.z_mas[i, :self.nv_vqs[i]] - self.z_mas[i + 1, :self.nv_vqs[i]]) < 0:
        #         print(f'check: master grid layer={i+1}, hsm={self.hsm[i+1]}, nv={self.nv_vqs[i+1]}')
        if ax is None:
            fig, ax = plt.subplots(figsize=[10, 5])
        for i in np.arange(nhm): 
            z_mas_ = self.z_mas[:, i]
            ax.plot(i * np.ones(self.nv_vqs[i]), z_mas_[np.isfinite(z_mas_)], 'k-', lw=0.3)
            if annotate:
                ax.text(
                    x=i, 
                    y=5, 
                    s=f"$\\theta_f={self.theta_f[i]:0.2f}$\n$\\theta_b={self.theta_b[i]:0.2f}$\n$hc={self.hc[i]:0.2f}$\nnv={self.nv_vqs[i]}",
                    fontsize=6            )        
        for k in np.arange(self.nv_vqs[-1]): 
            ax.plot(np.arange(nhm), self.z_mas[k,:], 'k-', lw=0.3)
        ax.set_xlim([-0.5, nhm - 0.5])
        ax.set_ylim([-self.hsm[-1], 0.5])
        ax.set_xlabel("Vertical Column Index")
        ax.set_ylabel("Depth [m]")    

    def plot_nv(self, hgrid: Hgrid, use_mesh: bool = False) -> None:
        """
        Plot a map of number of vertical levels (kbp) at nodes.

        Parameters
        ----------
        hgrid : Hgrid
        use_mesh : bool
            If True and elements are present, tries to render a triangulated field (quads split).
            Otherwise uses a scatter plot.
        """
        if self.kbp is None:
            raise RuntimeError("Run calc_z_cor() first to compute kbp.")
        plt.figure()
        tpc = plt.tricontourf(hgrid.triangulation, self.kbp, levels=np.unique(self.nv_vqs), shading="flat")
        plt.colorbar(tpc, label="# vertical levels (kbp)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Map of # vertical levels")
        plt.axis("equal")
        plt.grid(True, alpha=0.2)
        plt.show()

    # -------------------------
    # Sigma / z helpers
    # -------------------------
    def gen_vqs_S(self, hsm, N, hc=10.5, theta_f=0.0, theta_b=0.0):
        
        hc=min(hsm,hc)
        k_idx = np.arange(N, dtype=float)
        sigma = (k_idx) / (1.0 - N)  
        tf = theta_f
        tb = theta_b
        if tf == 0:
            cs = sigma
        else:
            cs = (1.0 - tb) * np.sinh(tf * sigma) / np.sinh(tf) + tb * (
                (np.tanh(tf * (sigma + 0.5)) - np.tanh(tf * 0.5)) / (2.0 * np.tanh(tf * 0.5))
            )
        z = self.etal * (1.0 + sigma) + hc * sigma + (hsm - hc) * cs
        return z

    def roms_vtransform2_vstretching4(self, hsm, N, hc=10.5, theta_f=0.0, theta_b=0.0):
        """
        Compute ROMS vertical coordinate z(k, ...) for:
        Vtransform = 2
        Vstretching = 4

        https://www.myroms.org/wiki/Vertical_S-coordinate

        Parameters
        ----------
        hsm : array-like
            Bathymetry (positive, meters). Can be scalar, 1D, or 2D.
        N : int
            Number of vertical rho-levels.
        hc : float
            Critical depth (Tcline) [m]. Can be any positive value for Vtransform=2. Taken as hc=min(hc,hsm)
        theta_f : float
            Surface stretching parameter ([0, 10] typical) ; this is theta_s in roms docs
        theta_b : float
            Bottom  stretching parameter ([0, 4] typical).

        Returns
        -------
        z : ndarray
            Depths (meters), shape:
            - (N, ...) for grid="rho"
            - (N+1, ...) for grid="w"
            where "..." is the broadcasted shape of h and zeta.
        s : ndarray
            Stretched s-coordinate (dimensionless), shape (N,) 
        Cs : ndarray
            Stretching function C(s), shape (N,) 

        Notes
        -----
        Vtransform=2:
        z0 = (hc*s + h*Cs) / (hc + h)
        z  = z0*(zeta + h) + zeta

        Vstretching=4 (double stretching):
        Surface:  Cs = (1 - cosh(theta_f*s)) / (cosh(theta_f) - 1)   if theta_f > 0
                    Cs = -s^2                                          if theta_f <= 0
        Bottom:   Cs = (exp(theta_b*Cs) - 1) / (1 - exp(-theta_b))   if theta_b > 0
        """
        hc = min(hc,hsm)
        h = np.asarray(hsm, dtype=float)
        zeta = np.asarray(self.etal, dtype=float)
        
        # k = np.arange(1, N + 1, dtype=float)
        # s = (k - N - 0.5) / N

        k = np.arange(N, dtype=float)
        s = (k) / (1.0 - N)  

        # --- Vstretching = 4: surface refinement (Eq. 8 in WikiROMS)
        if theta_f > 0.0:
            Cs = (1.0 - np.cosh(theta_f * s)) / (np.cosh(theta_f) - 1.0)
        else:
            Cs = -(s ** 2)

        # --- bottom refinement (Eq. 9 in WikiROMS)
        if theta_b > 0.0:
            Cs = (np.exp(theta_b * Cs) - 1.0) / (1.0 - np.exp(-theta_b))

        # --- Vtransform = 2 vertical transformation
        # Broadcast h, zeta against vertical s/Cs
        s3 = s[(slice(None),) + (None,) * h.ndim]
        Cs3 = Cs[(slice(None),) + (None,) * h.ndim]

        denom = (hc + h)
        if np.any(denom == 0.0):
            raise ValueError("hc + h must be nonzero everywhere.")

        z0 = (hc * s3 + h * Cs3) / denom
        z = z0 * (zeta + h) + zeta

        return z, s, Cs

class VgridTypeDispatch(Enum):

    LSC2 = LSC2
    SZ = SZ

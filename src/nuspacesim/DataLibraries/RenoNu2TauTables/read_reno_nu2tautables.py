import numpy as np
import math
import h5py

def extract_pexit_data(filename):
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    b = [(math.pi*float(l[0])/180.0) for l in data]
    le = [math.log10(float(l[1])) for l in data]
    p = [math.log10(float(l[-1])) for l in data]
    infile.close()
    return b, le, p

def extra_taudist_data(filename):
    bdeg = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0])
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    brad = math.pi*bdeg/180.0
    z = np.array([float(l[0]) for l in data])
    for l in data:
        del l[0]
    cv = np.array(data,float)
    infile.close()
    return z, brad, cv
    
def main():
    f = h5py.File('RenoNu2TauTables/nu2taudata.hdf5','w')
    pexitgrp = f.create_group('pexitdata')
    blist, lelist, plist = extract_pexit_data('RenoNu2TauTables/multi-efix.26')
    beta = np.array(blist)
    logenergy = np.array(lelist)
    pexitval = np.array(plist)
    buniq = np.unique(beta)
    leuniq = np.unique(logenergy)
    pexitarr = pexitval.reshape((leuniq.size,buniq.size))
    
    bdset = pexitgrp.create_dataset('BetaRad', data=buniq, dtype='f')
    ledset = pexitgrp.create_dataset('logNuEnergy', data=leuniq, dtype='f')
    pexitdset = pexitgrp.create_dataset('logPexit', data=pexitarr, dtype='f')

    for lognuenergy in np.arange(7.0,11.0,0.25):
        mygrpstring = 'TauEdist_grp_e{:02.0f}_{:02.0f}'.format(math.floor(lognuenergy),(lognuenergy - math.floor(lognuenergy))*100)
        tedistgrp = f.create_group(mygrpstring)
        
        myfilestring = 'RenoNu2TauTables/nu2tau-angleC-e{:02.0f}-{:02.0f}smx.dat'.format(math.floor(lognuenergy),(lognuenergy - math.floor(lognuenergy))*100)
        tauEfrac, tdbeta, cdfvalues = extra_taudist_data(myfilestring)

        tauEdset = tedistgrp.create_dataset('TauEFrac', data=tauEfrac, dtype='f')
        tdbdset = tedistgrp.create_dataset('BetaRad', data=tdbeta, dtype='f')
        cdfdset = tedistgrp.create_dataset('TauEDistCDF', data=cdfvalues, dtype='f')

if __name__ == "__main__":
    main ()

import csv
import matplotlib.pyplot as plt
import numpy


def autocorr1(x,lags):
    '''numpy.corrcoef, partial'''

    corr=[1. if l==0 else numpy.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return numpy.array(corr)

def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=numpy.mean(x)
    var=numpy.var(x)
    xp=x-mean
    corr=[1. if l==0 else numpy.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]

    return numpy.array(corr)

def autocorr3(x,lags):
    '''fft, pad 0s, non partial'''

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**numpy.ceil(numpy.log2(ext_size)).astype('int')

    xp=x-numpy.mean(x)
    var=numpy.var(x)

    # do fft and ifft
    cf=numpy.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=numpy.fft.ifft(sf).real
    corr=corr/var/n

    return corr[:len(lags)]

def autocorr4(x,lags):
    '''fft, don't pad 0s, non partial'''
    mean=x.mean()
    var=numpy.var(x)
    xp=x-mean

    cf=numpy.fft.fft(xp)
    sf=cf.conjugate()*cf
    corr=numpy.fft.ifft(sf).real/var/len(x)

    return corr[:len(lags)]

def autocorr5(x,lags):
    '''numpy.correlate, non partial'''
    mean=x.mean()
    var=numpy.var(x)
    xp=x-mean
    corr=numpy.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]

data = csv.reader(open('/home/martin/git_repositories/sherlockpipe/TIC305048087_all/1/lc_0.csv', 'r'))
time, flux, fluxerr = [], [], []
header = True
for row in data:
    if header:
        header = False
    else:
        time.append(row[0])
        flux.append(row[1])
        fluxerr.append(row[2])
flux = numpy.array(flux).astype('float')
fig,ax=plt.subplots()
lags = range(1, 100)
for funcii, labelii in zip([autocorr1, autocorr2, autocorr3, autocorr4,
        autocorr5], ['np.corrcoef, partial', 'manual, non-partial',
            'fft, pad 0s, non-partial', 'fft, no padding, non-partial',
            'np.correlate, non-partial']):
    cii=funcii(flux,lags)
    print(labelii)
    print(cii)
    ax.plot(lags,cii,label=labelii)
ax.set_xlabel('lag')
ax.set_ylabel('correlation coefficient')
ax.legend()
plt.show()
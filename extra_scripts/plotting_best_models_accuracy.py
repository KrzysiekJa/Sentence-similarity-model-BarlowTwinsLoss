import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


fsize = 15
tsize = 13
tdir = 'in'
major = 5.0
minor = 3.0
style = 'default'
lwidth = 0.5
lhandle = 3.0

plt.style.use(style)
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = lhandle


xsize = 8
ysize = 5
plt.figure( figsize=(xsize, ysize) )



########################################################################
# English
########################################################################
# sentence-transformers/nli-distilroberta-base-v2
nli_distilroberta_train_0_0005 = []
nli_distilroberta_validation_0_0005 = []
nli_distilroberta_test_0_0005 = 
nli_distilroberta_train_0_0001 = []
nli_distilroberta_validation_0_0001 = []
nli_distilroberta_test_0_0001 = 
# sentence-transformers/stsb-roberta-base-v2
stsb_roberta_train_0_005 = []
stsb_roberta_validation_0_005 = []
stsb_roberta_test_0_005 = 
stsb_roberta_train_0_0005 = []
stsb_roberta_validation_0_0005 = []
stsb_roberta_test_0_0005 = 


x = range(1, len(nli_distilroberta_train_0_0005)+1)
# scalling from <0,1> to range <0, 100>
nli_distilroberta_train_0_0005 = [i*100 for i in nli_distilroberta_train_0_0005]
nli_distilroberta_validation_0_0005 = [i*100 for i in nli_distilroberta_validation_0_0005]
nli_distilroberta_train_0_0001 = [i*100 for i in nli_distilroberta_train_0_0001]
nli_distilroberta_validation_0_0001 = [i*100 for i in nli_distilroberta_validation_0_0001]
stsb_roberta_train_0_005 = [i*100 for i in stsb_roberta_train_0_005]
stsb_roberta_validation_0_005 = [i*100 for i in stsb_roberta_validation_0_005]
stsb_roberta_train_0_0005 = [i*100 for i in stsb_roberta_train_0_0005]
stsb_roberta_validation_0_0005 = [i*100 for i in stsb_roberta_validation_0_0005]

plt.plot( x, nli_distilroberta_train_0_0005, '-o', label=r'NLI-DistilRoBERTa-base-v2, $\lambda$ = 0.0005, tren.', lw=1.1, ms=4 , c='#FF9A49' )
plt.plot( x, nli_distilroberta_validation_0_0005, '-D', label=r'NLI-DistilRoBERTa-base-v2, $\lambda$ = 0.0005, wal.', lw=0.8, ms=3.2, c='#FFCC66' )
plt.plot( x, nli_distilroberta_train_0_0001, '-o', label=r'NLI-DistilRoBERTa-base-v2, $\lambda$ = 0.0001, tren.', lw=1.1, ms=4 , c='#D21E05' )
plt.plot( x, nli_distilroberta_validation_0_0001, '-D', label=r'NLI-DistilRoBERTa-base-v2, $\lambda$ = 0.0001, wal.', lw=0.8, ms=3.2, c='#F39E9E' )
plt.plot( x, stsb_roberta_train_0_005, '-o', label=r'STS-B-RoBERTa-base-v2, $\lambda$ = 0.005, tren.', lw=1.1, ms=4 , c='#1F44A3' )
plt.plot( x, stsb_roberta_validation_0_005, '-D', label=r'STS-B-RoBERTa-base-v2, $\lambda$ = 0.005, wal.', lw=0.8, ms=3.2, c='#79C1E8' )
plt.plot( x, stsb_roberta_train_0_0005, '-o', label=r'STS-B-RoBERTa-base-v2, $\lambda$ = 0.0005, tren.', lw=1.1, ms=4 , c='#188977' )
plt.plot( x, stsb_roberta_validation_0_0005, '-D', label=r'STS-B-RoBERTa-base-v2, $\lambda$ = 0.0005, wal.', lw=0.8, ms=3.2, c='#6FC486' )


########################################################################
# Polski
########################################################################
# sdadas/polish-roberta-base-v1
pl_roberta_base_train_0_005 = []
pl_roberta_base_validation_0_005 = []
pl_roberta_base_test_0_005 = 
pl_roberta_base_train_0_0005 = []
pl_roberta_base_validation_0_0005 = []
pl_roberta_base_test_0_0005 = 
# sdadas/polish-roberta-large-v2
pl_roberta_large_train_0_0005 = []
pl_roberta_large_validation_0_0005 = []
pl_roberta_large_test_0_0005 = 
pl_roberta_large_train_0_0001 = []
pl_roberta_large_validation_0_0001 = []
pl_roberta_large_test_0_0001 = 


x = range(1, len(pl_roberta_base_train_0_005)+1)
# scalling from <0,1> to range <0, 100>
pl_roberta_base_train_0_005 = [i*100 for i in pl_roberta_base_train_0_005]
pl_roberta_base_validation_0_005 = [i*100 for i in pl_roberta_base_validation_0_005]
pl_roberta_base_train_0_0005 = [i*100 for i in pl_roberta_base_train_0_0005]
pl_roberta_base_validation_0_0005 = [i*100 for i in pl_roberta_base_validation_0_0005]
pl_roberta_large_train_0_0005 = [i*100 for i in pl_roberta_large_train_0_0005]
pl_roberta_large_validation_0_0005 = [i*100 for i in pl_roberta_large_validation_0_0005]
pl_roberta_large_train_0_0001 = [i*100 for i in pl_roberta_large_train_0_0001]
pl_roberta_large_validation_0_0001 = [i*100 for i in pl_roberta_large_validation_0_0001]

plt.plot( x, pl_roberta_base_train_0_005, '-o', label=r'Polish-RoBERTa-base-v1, $\lambda$ = 0.005, tren.', lw=1.1, ms=4 , c='#D21E05' )
plt.plot( x, pl_roberta_base_validation_0_005, '-D', label=r'Polish-RoBERTa-base-v1, $\lambda$ = 0.005, wal.', lw=0.8, ms=3.2, c='#F39E9E' )
plt.plot( x, pl_roberta_base_train_0_0005, '-o', label=r'Polish-RoBERTa-base-v1, $\lambda$ = 0.0005, tren.', lw=1.1, ms=4 , c='#FF9A49' )
plt.plot( x, pl_roberta_base_validation_0_0005, '-D', label=r'Polish-RoBERTa-base-v1, $\lambda$ = 0.0005, wal.', lw=0.8, ms=3.2, c='#FFCC66' )
plt.plot( x, pl_roberta_large_train_0_0005, '-o', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0005, tren.', lw=1.1, ms=4 , c='#1F44A3' )
plt.plot( x, pl_roberta_large_validation_0_0005, '-D', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0005, wal.', lw=0.8, ms=3.2, c='#79C1E8' )
plt.plot( x, pl_roberta_large_train_0_0001, '-o', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0001, tren.', lw=1.1, ms=4 , c='#188977' )
plt.plot( x, pl_roberta_large_validation_0_0001, '-D', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0001, wal.', lw=0.8, ms=3.2, c='#6FC486' )



title = 'Liczba epok: 20'

ax = plt.gca()
ax.xaxis.set_minor_locator( MultipleLocator(1) )
ax.yaxis.set_minor_locator( MultipleLocator(.5) )

plt.xlabel( 'Liczba epok', labelpad=10 )
plt.xticks( x )
plt.ylabel( r'$\rho$ Spearmana', labelpad=20 )
plt.title( title, pad=10 )
plt.legend( loc='best' )
    
plt.show()


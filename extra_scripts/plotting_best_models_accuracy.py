import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


fsize = 13
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
# # sentence-transformers/nli-distilroberta-base-v2
# nli_distilroberta_train_0_0005 = [0.8303175366840527, 0.8336455299106837, 0.8357068339445913, 0.8439274617960768, 0.8502022045198147, 0.8543921318710674, 0.8673271579466045, 0.8700856673281744, 0.8781370760555198, 0.8888676909526448, 0.8907283846102279, 0.8957870862961587, 0.9040399483624303, 0.9063215753279151, 0.9075859883706794, 0.9119670729278007, 0.9122423987454527, 0.9161502349805357, 0.9182914787916957, 0.9213677745201215, 0.9239196112866034, 0.9256179828934833, 0.9277005710826204, 0.9270129962411983, 0.9307955030654894, 0.9312648572880288, 0.9326828515164688, 0.9316375590448767, 0.935116318923144, 0.9344102693739369, 0.9341055054798542, 0.9357827250738682, 0.9368529275146734, 0.9372354846558538, 0.935811686167878, 0.9374150083640986, 0.9389101502334798, 0.938566393036197, 0.9385411128729726, 0.939094663091395, 0.9388332239278099, 0.9390023538139577, 0.938482159969893, 0.9397792962910458, 0.9393030267972322, 0.9397421377386097, 0.9397024072920657, 0.939599515239422, 0.9397893510374505, 0.9397118114436502]
# nli_distilroberta_validation_0_0005 = [0.8577507111600023, 0.8559071262578316, 0.8561741193246485, 0.8574369084523406, 0.8579770013517549, 0.860933958908397, 0.8584446297883491, 0.8618347299926621, 0.8555378662165243, 0.8642748419968943, 0.8596090113748195, 0.8561161530924746, 0.8603893749026805, 0.8572363300221043, 0.8608511243316589, 0.8636652437114354, 0.8631942825323274, 0.8635588379440533, 0.8649452702746674, 0.8622583495020885, 0.8638395000594495, 0.8637603825671272, 0.8588839450381446, 0.8621111572296909, 0.8614478987648517, 0.8625358425761517, 0.8621235326830712, 0.8632553822186212, 0.8638786744637084, 0.8621437889248083, 0.8627784904304383, 0.8623789207596732, 0.8641147315661157, 0.8634723859270965, 0.862954337526881, 0.8615678935095831, 0.8620607408247912, 0.862084141196812, 0.8634116099622398, 0.8646727853320174, 0.8627766078668614, 0.8634628841411798, 0.8641309539972397, 0.8637665000090713, 0.8622661199700843, 0.8627839690819058, 0.8625091130203387, 0.8623891271724107, 0.8627329333366909, 0.8628174032686804]
# nli_distilroberta_test_0_0005 = 0.8386420634011204
# # sentence-transformers/stsb-roberta-base-v2
# stsb_roberta_train_0_005 = [0.964637156966342, 0.9585884780314912, 0.9522400482329919, 0.9455932071175998, 0.9358344328274175, 0.9407592020980148, 0.9417795198399379, 0.9467208748539151, 0.9466054800196424, 0.9514481077934723, 0.9512414693379497, 0.9550489209867539, 0.9563113197895261, 0.9600187683901504, 0.958501411131567, 0.9604974854505015, 0.9617205670183588, 0.9635758524797025, 0.9629813690947063, 0.9658903401703445, 0.9663045752036659, 0.966597382004504, 0.9671762944762183, 0.9672066755241768, 0.9672695757453277, 0.9690435099428448, 0.9686290906419391, 0.969349357215047, 0.9686576237716018, 0.9706472866185275, 0.9706653711819896, 0.9701287572583078, 0.9702226810635103, 0.9715865582854847, 0.9721997697254493, 0.9714366872837277, 0.9707672048024159, 0.9715959652156404, 0.9732528771899147, 0.9720518858316224, 0.9733448770847035, 0.9721026394654336, 0.9730705270659933, 0.9729843900549113, 0.9733215684120511, 0.9734534527017559, 0.9734889617237884, 0.9732097796585817, 0.9732606035235918, 0.9733131588199101]
# stsb_roberta_validation_0_005 = [0.8659310673391614, 0.8732484887590987, 0.8579013304810282, 0.8618902691768978, 0.8640557012762679, 0.8668567797217774, 0.8609596510969459, 0.8617314719142852, 0.8642540092423037, 0.8634032666623146, 0.860688332404368, 0.8650365418108077, 0.8651420917057128, 0.8650643087338845, 0.8667354380010277, 0.8614126496302056, 0.8672127870848997, 0.8640874308355314, 0.8652016842735287, 0.8635313147933867, 0.863650174306018, 0.8610926565268128, 0.8666139023299659, 0.8640425820501323, 0.8615732529438868, 0.8604543872782308, 0.8604042717101141, 0.8633891206228561, 0.8622378530466625, 0.8599274831248493, 0.8593433688427423, 0.8603408481788003, 0.8606069533446488, 0.8593749827435629, 0.8561134022008964, 0.8565910207296425, 0.8550052377350728, 0.8574450899526486, 0.8557646314975725, 0.8591697530645652, 0.8540488402778996, 0.8572073922797141, 0.8572393495971419, 0.8576653694640964, 0.8587290427960136, 0.8584355586077119, 0.8567686268523623, 0.8569798653043837, 0.8572641414291976, 0.8571009598219224]
# stsb_roberta_test_0_005 = 0.8335633502210283
#
#
# x = range(1, len(nli_distilroberta_train_0_0005)+1)
#
# plt.plot( x, nli_distilroberta_train_0_0005, '-o', label=r'NLI-DistilRoBERTa-base-v2, $\lambda$ = 0.0005, tren.', lw=1.1, ms=4 , c='#FF9A49' )
# plt.plot( x, nli_distilroberta_validation_0_0005, '-D', label=r'NLI-DistilRoBERTa-base-v2, $\lambda$ = 0.0005, wal.', lw=0.8, ms=3.2, c='#FFCC66' )
# plt.plot( x, stsb_roberta_train_0_005, '-o', label=r'STS-B-RoBERTa-base-v2, $\lambda$ = 0.005, tren.', lw=1.1, ms=4 , c='#1F44A3' )
# plt.plot( x, stsb_roberta_validation_0_005, '-D', label=r'STS-B-RoBERTa-base-v2, $\lambda$ = 0.005, wal.', lw=0.8, ms=3.2, c='#79C1E8' )


########################################################################
# Polski
########################################################################
# sdadas/polish-roberta-base-v1
pl_roberta_base_train_0_005 = [0.8653475920360892, 0.861162088074763, 0.8734976146413449, 0.8799586093661269, 0.8728039447242502, 0.8872527138090689, 0.8907389721786881, 0.8943777334366027, 0.9070955567162962, 0.9121066422632955, 0.9132069615857047, 0.9182152998889086, 0.9230737424062375, 0.922456913709964, 0.9234748143736674, 0.928613102006037, 0.9271339604669211, 0.9301132112331318, 0.9297194423790966, 0.9298326311218181, 0.9292094416969081, 0.9327691742463974, 0.9318465858704142, 0.934639608224221, 0.9362000712305136, 0.9353851798330298, 0.9355504873570614, 0.9367900112614516, 0.9373846603201716, 0.9377687423329776, 0.9366589701572887, 0.9386669245162796, 0.9380899894059398, 0.9370355147993905, 0.94006220154232, 0.9383556435128377, 0.9394682058698667, 0.9387753825381088, 0.9406365012955337, 0.9409857793113499, 0.9406807095340562, 0.9414398575914866, 0.9406510278776264, 0.9402948744624454, 0.9412805193650042, 0.9418375440940977, 0.9405735391803579, 0.9418595433784887, 0.9415702590116828, 0.9415805793389158]
pl_roberta_base_validation_0_005 = [0.8326043580834966, 0.8398429469465981, 0.8430074775091032, 0.8436687773291996, 0.8587485294994366, 0.8549146164592031, 0.860514936256529, 0.8665693981800061, 0.8605004015499886, 0.8600224924218594, 0.8617930402212509, 0.8633605713689969, 0.8611694438287747, 0.8617602168888053, 0.8592205848595563, 0.8555760805132716, 0.8623576648867065, 0.8687152909918829, 0.8650529555286153, 0.8601208568402142, 0.8608680275657797, 0.8595597280121674, 0.8605548099187816, 0.8665077752474338, 0.8643025001236058, 0.8645060915941545, 0.86153722234754, 0.8609125467032461, 0.8627993486929658, 0.863962829076084, 0.8612016102253315, 0.8656842357247914, 0.863374050285716, 0.8640327106312589, 0.864069018070115, 0.8648218900607157, 0.8677427326014571, 0.8679023914844398, 0.8671743188236428, 0.8681474872259652, 0.8674835713933116, 0.8680211678393408, 0.8660220532746267, 0.8660846968040262, 0.8663978557960341, 0.8693854410785455, 0.866737726270522, 0.8666656392877204, 0.8679083508314315, 0.86841717113933]
pl_roberta_base_test_0_005 = 0.8667398329601338
# sdadas/polish-roberta-large-v2
pl_roberta_large_train_0_0005 = [0.8184895389016653, 0.877416381020692, 0.8900787592624376, 0.9013943860251248, 0.8799554301373647, 0.8895029348994716, 0.9021894475933067, 0.906708540159224, 0.9087664682174251, 0.9212157573907199, 0.9225294904804626, 0.9253604179077198, 0.930322898704485, 0.9328077947442818, 0.9321313303312684, 0.9352877689262133, 0.9346678633910595, 0.9354947482419214, 0.9352995405787452, 0.9383100587916599, 0.9373535887408538, 0.9390375213859137, 0.9396815453283562, 0.9396682138556809, 0.9424425798808613, 0.9404438821833373, 0.9424757994667999, 0.9398050037427679, 0.9437549914719009, 0.9416154638996652, 0.9441472402868357, 0.9450265359781925, 0.9436871740016147, 0.945687728652095, 0.9432558083571192, 0.9463952692055507, 0.9442972203343926, 0.9463423945167085, 0.945664149729665, 0.9460054188160851, 0.9461475543609947, 0.9462138055982057, 0.9452017130280309, 0.9450374178874401, 0.9455556282387536, 0.9455434638432888, 0.9452808147475411, 0.9452638343307087, 0.9451779648335703, 0.9452338799321952]
pl_roberta_large_validation_0_0005 = [0.867714273200273, 0.8771641786511803, 0.8650703291366754, 0.8776417827733611, 0.8771127382248849, 0.8805170034628226, 0.8840043593596998, 0.8854837789813556, 0.8804322118091709, 0.8860755726382136, 0.8824852068477675, 0.8801008815012379, 0.888517192179048, 0.8884894014447506, 0.8879021828771185, 0.8836359942910338, 0.8847321622075426, 0.8891827972754157, 0.8857971022073357, 0.8858639337030266, 0.8896427345145862, 0.8851782920620496, 0.8870562488792657, 0.883038804375089, 0.8831143050783129, 0.8817142691202239, 0.8820871036991338, 0.8827151344088596, 0.8810393378804344, 0.8881010350244694, 0.8777393964078433, 0.8831182701556419, 0.8866060600994456, 0.8845376974534495, 0.8847782298367473, 0.8831209917471813, 0.8800599637801618, 0.8835726110204515, 0.8837956642180138, 0.8819428945405363, 0.8838217311826037, 0.8842672510252139, 0.8817444999021082, 0.8825962524749766, 0.8822850760216332, 0.8816297824725206, 0.8799626668826263, 0.8795294176639397, 0.8799032728396768, 0.8802350020015423]
pl_roberta_large_test_0_0005 = 0.872364671284727
# pl_roberta_large_train_0_0001 = [0.8522243874090533, 0.8699841072818792, 0.8971116204257206, 0.9014384561917205, 0.8772989347192411, 0.9000323161268785, 0.908169039658126, 0.916489865105765, 0.9172880495765491, 0.9198319351708246, 0.924150734848626, 0.9246402635833612, 0.92578097082686, 0.9288299100949478, 0.9326769892981053, 0.9310481469753583, 0.9335059759808099, 0.9334847408291327, 0.9354757250467922, 0.9384261488832484, 0.9369822578296556, 0.938067303504124, 0.9383505089718488, 0.9402822288291195, 0.940874532582231, 0.9374104449115087, 0.941832607182709, 0.943154287944766, 0.9417017293183695, 0.9426191356063978, 0.9428961295592844, 0.9396117394110611, 0.9376058385973012, 0.9372675521733601, 0.9369908396557223, 0.9368320740379087, 0.9353322324618764, 0.9349346830620155, 0.9306286550930882, 0.9299518883905253, 0.9294003988724743, 0.9312309802944962, 0.9293442279232524, 0.9298746343121621, 0.9287571549026293, 0.9277001215815394, 0.9288780048996383, 0.9274703759213356, 0.9271992670671532, 0.9277511549823376]
# pl_roberta_large_validation_0_0001 = [0.885980680595271, 0.8764521774576636, 0.8725328744828796, 0.8862361817320357, 0.8881467037997413, 0.8941935171125126, 0.8886971691505927, 0.8963002635840136, 0.8881399819378787, 0.8935654981337846, 0.895841018473726, 0.8930455568397679, 0.8949093895353742, 0.8940601708580772, 0.8915291376503731, 0.896019775421477, 0.8934011116586018, 0.8906717893914518, 0.8896699504299808, 0.8902911302368661, 0.892153883680655, 0.8880188593833749, 0.8940958682850365, 0.8917581619246182, 0.8939664049909439, 0.8919706454916622, 0.8944663801263395, 0.8953842720660157, 0.8954180925332926, 0.8977488769996386, 0.896339949550298, 0.8934466161999016, 0.8930621327399632, 0.8905839711403122, 0.8877659390661734, 0.8886600522728736, 0.8888444283686738, 0.884664838009935, 0.8868160801569065, 0.8856908310963187, 0.8845532527568173, 0.8808074395116757, 0.882376108566229, 0.8815944487064996, 0.8778153194269958, 0.8785488352708718, 0.8767471064787564, 0.8777926082148388, 0.8761561105297634, 0.8762259334299483]
# pl_roberta_large_test_0_0001 = 0.8529795004337815


x = range(1, len(pl_roberta_base_train_0_005)+1)

plt.plot( x, pl_roberta_base_train_0_005, '-o', label=r'Polish-RoBERTa-base-v1, $\lambda$ = 0.005, tren.', lw=1.1, ms=4 , c='#D21E05' )
plt.plot( x, pl_roberta_base_validation_0_005, '-D', label=r'Polish-RoBERTa-base-v1, $\lambda$ = 0.005, wal.', lw=0.8, ms=3.2, c='#F39E9E' )
plt.plot( x, pl_roberta_large_train_0_0005, '-o', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0005, tren.', lw=1.1, ms=4 , c='#1F44A3' )
plt.plot( x, pl_roberta_large_validation_0_0005, '-D', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0005, wal.', lw=0.8, ms=3.2, c='#79C1E8' )
# plt.plot( x, pl_roberta_large_train_0_0001, '-o', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0001, tren.', lw=1.1, ms=4 , c='#188977' )
# plt.plot( x, pl_roberta_large_validation_0_0001, '-D', label=r'Polish-RoBERTa-large-v2, $\lambda$ = 0.0001, wal.', lw=0.8, ms=3.2, c='#6FC486' )



title = 'Liczba epok: 50'

ax = plt.gca()
ax.xaxis.set_minor_locator( MultipleLocator(1) )
ax.yaxis.set_minor_locator( MultipleLocator(.005) )

plt.xlabel( 'Liczba epok', labelpad=10 )
plt.xticks( x )
plt.ylabel( r'$\rho$ Spearmana', labelpad=20 )
plt.title( title, pad=10 )
plt.legend( loc='best' )

plt.show()

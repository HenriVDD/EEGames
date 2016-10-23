import glob
import numpy as np
import mne
import glob
import re
from scipy.signal import spectrogram
from scipy.ndimage.filters import gaussian_filter

def _parse_tal_channel(tal_channel_data):
	"""Parse time-stamped annotation lists (TALs) in stim_channel.
	Parameters
	----------
	tal_channel_data : ndarray, shape = [n_samples]
	channel data in EDF+ TAL format
	Returns
	-------
	events : list
	List of events. Each event contains [start, duration, annotation].
	References
	----------
	http://www.edfplus.info/specs/edfplus.html#tal
	"""
	# convert tal_channel to an ascii string
	tals = bytearray()
	for s in tal_channel_data:
		i = int(s)
		tals.extend(np.uint8([i % 256, i // 256]))
	regex_tal = '([+-]\d+\.?\d*)(\x15(\d+\.?\d*))?(\x14.*?)\x14\x00'
	# use of latin-1 because characters are only encoded for the first 256
	# code points and utf-8 can triggers an "invalid continuation byte" error
	tal_list = re.findall(regex_tal, tals.decode('latin-1'))
	events = []
	for ev in tal_list:
		onset = float(ev[0])
		duration = float(ev[2]) if ev[2] else 0
		for annotation in ev[3].split('\x14')[1:]:
			if annotation:
				events.append([onset, duration, annotation])
	return events

def loaddata(path):
    raw   = mne.io.read_raw_edf(path, preload=True, stim_channel=None)
    raw.set_eeg_reference()
    data  = raw._data[:-1]
    N     = data.shape[1]
    a     = _parse_tal_channel(raw._data[-1])
    dt    = (a[-1][0] + a[-1][1])/N
    label = np.zeros(N)
    for i in a:
        label[int(i[0]/dt):int((i[0]+i[1])/dt)] = int(i[2][-1])
    return data,label

def expspace(N,pmin,pmax):
    N = float(N)
    return pmin * np.exp((np.arange(N)/(N-1))*np.log(pmax/pmin))


################################################################################
listfiles = np.sort(glob.glob("*/*.edf"))
S = np.array([file[-15:-12] for file in listfiles]).astype(int)
Stim = np.array([file[-6:-4] for file in listfiles]).astype(int)
select = np.array([stim in [3,4,7,8] for stim in Stim])
files = listfiles[select]

all_data  = []
all_label = []

for ifile in files:
    data_, label = loaddata(ifile)
    mem = label[0]; begin = 0; delta=0
    for idx in range(len(label)):
    	if label[idx] == mem:
    		delta = delta + 1
    	else:
    		if mem != 2:
         		all_data.append(data[:, begin:(begin + delta)])
    			all_label.append(label[begin])
    		begin = begin + delta; delta = 1
    		mem   = label[idx]
    		assert(idx == begin)

all_spectrogram = np.zeros([len(all_label), 64 * 50])

for idx in range(len(all_data)):
	data         = all_data[idx]
	spectrogram_ = np.zeros([64, 50])
	for channel_idx in range(64):
		f,t,Sxx        = spectrogram(data[channel_idx], 160, nperseg=len(data[channel_idx]))
		spectre        = Sxx[(f < 30) * (f > 5)]
		spectre_smooth = gaussian_filter(spectre, spectre.shape[0]/25.)
		spectre_final  = np.interp(expspace(50,.1,len(spectre)-1), np.arange(len(spectre_smooth)), spectre_smooth[:,0])
		spectrogram_[channel_idx] = spectre_final
	all_spectrogram[idx] = spectrogram_.ravel()

all_label = np.asarray(all_label)

from sklearn import svm
clf        = svm.SVC()

all_spectrogram = (all_spectrogram - np.mean(all_spectrogram))/np.std(all_spectrogram) #(np.max(all_spectrogram) - np.min(all_spectrogram))

index_0 = np.where(all_label==0)[0]
index_1 = np.where(all_label==1)[0]

train_data = all_spectrogram[np.concatenate((index_0[:2000], index_1[:2000]))]
train_label = all_label[np.concatenate((index_0[:2000], index_1[:2000]))]

clf.fit(train_data, train_label)

test_data = all_spectrogram[int(len(all_spectrogram)*.6):int(len(all_spectrogram)*.8)]
test_label = all_label[int(len(all_spectrogram)*.6):int(len(all_spectrogram)*.8)]

prediction = clf.predict(test_data)
print 1 - sum(abs(prediction - test_label))*1./len(test_label)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(train_data, train_label)
prediction = rfc.predict(test_data)
print 1 - sum(abs(prediction - test_label))*1./len(test_label)

# plt.plot(f[(f < 30)*(f>5)], spectre_smooth)
# plt.plot(f[(f < 30)*(f>5)][::15], spectre_smooth[::15],'o')








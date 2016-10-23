import glob
import numpy as np
import mne
import glob
import re


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


################################################################################
listfiles = np.sort(glob.glob("eegmmidb/*/*.edf"))
S = np.array([file[-15:-12] for file in listfiles]).astype(int)
Stim = np.array([file[-6:-4] for file in listfiles]).astype(int)
select = np.array([stim in [3,4,7,8,11,12] for stim in Stim])
files = listfiles[select]


all_data  = []
all_label = []

for ifile in files:
    data, label = loaddata(ifile)
    all_data.append(data)
    all_label.append(label)

import mne
import numpy as np

path='S001R05.edf'
path_event=path+'.event'
raw = mne.io.read_raw_edf(path, preload=True, stim_channel=None)
# raw_event = mne.io.read_raw_edf(path_event, preload=True, stim_channel=None)


plt.plot(raw._data[-1])

nb_points = raw._data[0].size
raw.set_eeg_reference()

# data  = raw._data - raw._data[0]
label_char = _parse_tal_channel(raw._data[-1])
label      = np.zeros(nb_points)
dt         = (label_char[-1][0] + label_char[-1][1])/nb_points
for idx in range(len(label_char)):



plt.figure(figsize=(15,9))

plt.subplot(3,2,1)
plt.plot(data[1], linewidth=2); 

plt.subplot(3,2,2)
plt.plot(data[10], linewidth=2); 

plt.subplot(3,2,3)
plt.plot(data[20], linewidth=2); 

plt.subplot(3,2,4)
plt.plot(data[30], linewidth=2); 

plt.subplot(3,2,5)
plt.plot(data[40], linewidth=2); 

plt.subplot(3,2,6)
plt.plot(data[50], linewidth=2);

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
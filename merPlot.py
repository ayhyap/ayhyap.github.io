import re
import numpy as np
from PIL import Image, ImageDraw
import io
import base64

import asyncio
import js
from js import document, FileReader

from pyodide.ffi import create_proxy

notetype2channel = {
	0: None,
	1: 0,  # tap
	2: 0,  # tap bonus
	3: 1,  # forward snap
	4: 2,  # backward snap
	5: 3,  # swipe CW
	6: 3,  # swipe CW bonus
	7: 4,  # swipe CCW
	8: 4,  # swipe CCW bonus
	9: 0,  # hold start
	10: 13,  # hold segment
	11: 13,  # hold end
	12: None,  # mask add
	13: None,  # mask remove
	14: None,  # end of chart
	15: None,  # same-time guide line
	16: 5,  # chain
	17: None,  # ?
	18: None,  # ?
	19: None,  # ?
	20: 0,  # tap R
	21: 1,  # forward snap R
	22: 2,  # backward snap R
	23: 3,  # swipe CW R
	24: 4,  # swipe CCW R
	25: 0,  # hold start R
	26: 5,  # chain R
	27: None
}

CHANNEL_TAP = 0
CHANNEL_SNAP_FORWARD = 1
CHANNEL_SNAP_BACKWARD = 2
CHANNEL_SWIPE_CW = 3
CHANNEL_SWIPE_CCW = 4
CHANNEL_CHAIN = 5
CHANNEL_TAP_OVERLAPPING = 6
CHANNEL_WINDOW_TAP_SAFE = 7
CHANNEL_WINDOW_SNAP_FORWARD_SAFE = 8
CHANNEL_WINDOW_SNAP_BACKWARD_SAFE = 9
CHANNEL_WINDOW_SWIPE_CW_SAFE = 10
CHANNEL_WINDOW_SWIPE_CCW_SAFE = 11
CHANNEL_WINDOW_CHAIN_SAFE = 12
CHANNEL_HOLD = 13
CHANNEL_HOLD_OVERLAPPING = 14
CHANNEL_WINDOW_TAP_UNSAFE = 15
CHANNEL_WINDOW_SNAP_FORWARD_UNSAFE = 16
CHANNEL_WINDOW_SNAP_BACKWARD_UNSAFE = 17
CHANNEL_WINDOW_SWIPE_CW_UNSAFE = 18
CHANNEL_WINDOW_SWIPE_CCW_UNSAFE = 19
CHANNEL_VERTICAL_INDICATOR = 20

SAFE_WINDOW_CHANNELS = [CHANNEL_WINDOW_TAP_SAFE, CHANNEL_WINDOW_SNAP_FORWARD_SAFE, CHANNEL_WINDOW_SNAP_BACKWARD_SAFE,
						CHANNEL_WINDOW_SWIPE_CW_SAFE, CHANNEL_WINDOW_SWIPE_CCW_SAFE, CHANNEL_WINDOW_CHAIN_SAFE]
UNSAFE_WINDOW_CHANNELS = [CHANNEL_WINDOW_TAP_UNSAFE, CHANNEL_WINDOW_SNAP_FORWARD_UNSAFE,
						  CHANNEL_WINDOW_SNAP_BACKWARD_UNSAFE, CHANNEL_WINDOW_SWIPE_CW_UNSAFE,
						  CHANNEL_WINDOW_SWIPE_CCW_UNSAFE]

LINE_TYPE_NOTE = 1
LINE_TYPE_BPM = 2
LINE_TYPE_TIMESIG = 3
LINE_TYPE_HISPEED = 5
LINE_TYPE_REVERSE_START = 6
LINE_TYPE_REVERSE_END = 7
LINE_TYPE_REVERSE_MIDDLE = 8  # rename me
LINE_TYPE_STOP_START = 9
LINE_TYPE_STOP_END = 10

NOTE_TYPE_TAP = 1
NOTE_TYPE_TAP_BONUS = 2
NOTE_TYPE_TAP_R = 20
NOTE_TYPE_CHAIN = 16
NOTE_TYPE_CHAIN_R = 26
NOTE_TYPE_SNAP_FORWARD = 3
NOTE_TYPE_SNAP_FORWARD_R = 21
NOTE_TYPE_SNAP_BACKWARD = 4
NOTE_TYPE_SNAP_BACKWARD_R = 22
NOTE_TYPE_SWIPE_CW = 5
NOTE_TYPE_SWIPE_CW_BONUS = 6
NOTE_TYPE_SWIPE_CW_R = 23
NOTE_TYPE_SWIPE_CCW = 7
NOTE_TYPE_SWIPE_CCW_BONUS = 8
NOTE_TYPE_SWIPE_CCW_R = 24
NOTE_TYPE_HOLD_START = 9
NOTE_TYPE_HOLD_START_R = 25
NOTE_TYPE_HOLD_SEGMENT = 10
NOTE_TYPE_HOLD_END = 11
NOTE_TYPE_MASK_ADD = 12
NOTE_TYPE_MASK_REMOVE = 13
NOTE_TYPE_END_OF_CHART = 14

DURATION_OF_1_FRAME = 1 / 60

# kinda hacky, smuggling constants into timing window tuples but whatevs
TIMING_WINDOWS = {
	CHANNEL_TAP: (12, 6, 6, 12, CHANNEL_WINDOW_TAP_SAFE, CHANNEL_WINDOW_TAP_UNSAFE),
	CHANNEL_SNAP_FORWARD: (20, 10, 14, 20, CHANNEL_WINDOW_SNAP_FORWARD_SAFE, CHANNEL_WINDOW_SNAP_FORWARD_UNSAFE),
	CHANNEL_SNAP_BACKWARD: (20, 14, 10, 20, CHANNEL_WINDOW_SNAP_BACKWARD_SAFE, CHANNEL_WINDOW_SNAP_BACKWARD_UNSAFE),
	CHANNEL_SWIPE_CW: (20, 10, 10, 20, CHANNEL_WINDOW_SWIPE_CW_SAFE, CHANNEL_WINDOW_SWIPE_CW_UNSAFE),
	CHANNEL_SWIPE_CCW: (20, 10, 10, 20, CHANNEL_WINDOW_SWIPE_CCW_SAFE, CHANNEL_WINDOW_SWIPE_CCW_UNSAFE),
	CHANNEL_CHAIN: (8, 8, 8, 8, CHANNEL_WINDOW_CHAIN_SAFE, CHANNEL_WINDOW_CHAIN_SAFE)
}


class Note:
	def __init__(self, position, size, note_type=None, channel=None, timestamp=None, index=None):
		assert not (note_type is None and channel is None)
		self.note_type = note_type
		self.channel = channel if channel is not None else notetype2channel[note_type]

		self.position = position
		self.size = size

		assert not (timestamp is None and index is None)
		self.timestamp = timestamp
		self.index = index
		if self.index is not None:
			timing_window = TIMING_WINDOWS[self.channel]
			self.early_window = self.index - timing_window[0]
			self.late_window = self.index + timing_window[3]
		else:
			self.early_window = self.late_window = None

	def __str__(self):
		return "Note({}, {}, {}, {}, {}, {})".format(self.note_type, self.channel, self.position, self.size,
													 self.timestamp, self.index)

	def __repr__(self):
		return "Note({}, {}, {}, {}, {}, {})".format(self.note_type, self.channel, self.position, self.size,
													 self.timestamp, self.index)


class Hold:
	def __init__(self):
		self.notes = []

		# for interpolating notes
		self.baked = False
		self.timestamps = []
		self.positions = []
		self.sizes = []

	def add_note(self, note: Note):
		self.notes.append(note)
		self.baked = False

	def bake_notes(self):
		if self.baked:
			return
		timestamps = [note.timestamp for note in self.notes]
		sorted_idx = np.argsort(timestamps)
		self.notes = np.array(self.notes)[sorted_idx]
		self.timestamps = np.array(timestamps)[sorted_idx]
		self.positions = np.array([note.position for note in self.notes])
		self.sizes = np.array([note.size for note in self.notes])
		self.baked = True


# TODO: chart validation resolve duplicate bpm/timesig on same tick
class Conductor:
	def __init__(self, sorted_timing_events):
		# sorted timing events are one of the following:
		# (measure, tick, 'bpm', float(split[3]))
		# (measure, tick, 'timesig', (int(split[3]), int(split[4])))

		assert sorted_timing_events[0][0] == sorted_timing_events[0][1] == 0
		assert sorted_timing_events[1][0] == sorted_timing_events[1][1] == 0

		# record stuff information at each timing event
		# (measure, tick, timestamp, bpm, timesig)
		bpm = None
		timesig = None
		for timing_event in sorted_timing_events[:2]:
			measure, tick, event_type, value = timing_event
			if event_type == 'bpm':
				bpm = value
			elif event_type == 'timesig':
				timesig = value
		self.sorted_timing_events = [{'measure': 0,
									  'tick': 0,
									  'timestamp': 0,
									  'pseudo_timestamp': 0,
									  'bpm': bpm,
									  'timesig': timesig}]

		for timing_event in sorted_timing_events[2:]:
			measure, tick, event_type, value = timing_event

			delta_measures = measure - self.sorted_timing_events[-1]['measure'] + (
						tick - self.sorted_timing_events[-1]['tick']) / 1920
			delta_beats = delta_measures * timesig[0]
			crotchet_fraction = 4 / timesig[1]
			delta_time = delta_beats / bpm * 60 * crotchet_fraction
			if event_type == 'bpm':
				bpm = value
			elif event_type == 'timesig':
				timesig = value
			self.sorted_timing_events.append({'measure': measure,
											  'tick': tick,
											  'timestamp': self.sorted_timing_events[-1]['timestamp'] + delta_time,
											  'pseudo_timestamp': measure * 10_000 + tick,
											  'bpm': bpm,
											  'timesig': timesig})

	# returns timestamp of given measure and tick in seconds
	def get_timestamp(self, measure, tick):
		pseudo_timestamp = measure * 10_000 + tick
		for timing_event in self.sorted_timing_events[::-1]:
			if pseudo_timestamp >= timing_event['pseudo_timestamp']:
				delta_measures = measure - timing_event['measure'] + (tick - timing_event['tick']) / 1920
				delta_beats = delta_measures * timing_event['timesig'][0]
				crotchet_fraction = 4 / timing_event['timesig'][1]
				delta_time = delta_beats / timing_event['bpm'] * 60 * crotchet_fraction
				return timing_event['timestamp'] + delta_time
		raise ValueError('Invalid timestamp {} {}'.format(measure, tick))


def parse_mer(lines):
	linse = lines.split('\n')
	is_body = False
	starting_bpm = None
	starting_timesig = None
	end_measure = -1
	timing_events = []
	_timing_event_timestamps = []
	note_lines = []
	_note_timestamps = []
	for line in lines:
		if line.startswith('#BODY'):
			is_body = True
			continue
		if not is_body:
			continue
		split = re.split(r'\s+', line.strip())
		measure = int(split[0])
		end_measure = max(end_measure, measure)
		tick = int(split[1])
		line_type = int(split[2])
		_timestamp = measure * 10_000 + tick
		if line_type == LINE_TYPE_BPM:
			timing_events.append((measure, tick, 'bpm', float(split[3])))
			_timing_event_timestamps.append(_timestamp)
			if starting_bpm is None:
				assert measure == tick == 0, 'no starting bpm!'
				starting_bpm = float(split[3])
		elif line_type == LINE_TYPE_TIMESIG:
			timing_events.append((measure, tick, 'timesig', (int(split[3]), int(split[4]))))
			_timing_event_timestamps.append(_timestamp)
			if starting_timesig is None:
				assert measure == tick == 0, 'no starting timesig!'
				starting_timesig = (int(split[3]), int(split[4]))
		elif line_type == LINE_TYPE_NOTE:
			note_lines.append(line)
			_note_timestamps.append(_timestamp)
		else:
			continue

	_idx = np.argsort(_timing_event_timestamps)
	sorted_timing_events = np.array(timing_events, dtype='object')[_idx]
	conductor = Conductor(sorted_timing_events)

	_idx = np.argsort(_note_timestamps)
	sorted_note_lines = np.array(note_lines, dtype='object')[_idx]
	notes = []
	holds = {}
	for line in sorted_note_lines:
		# measure	linetyp	lineno	size	next_hold_lineno
		# 	tick	notetyp	pos		render
		# 0	1	2	3	4	5	6	7	8
		#
		# 4	0	1	9	3	37	10	1	4
		# 4	137	1	10	4	36	10	0	5
		# 4	274	1	10	5	35	10	0	6
		# 4	411	1	10	6	34	10	0	7
		# 4	548	1	10	7	33	10	0	8
		# 4	685	1	10	8	32	10	0	9
		# 4	822	1	10	9	31	10	0	10
		split = re.split(r'\s+', line.strip())
		measure = int(split[0])
		end_measure = max(end_measure, measure)
		tick = int(split[1])
		line_type = int(split[2])
		# note
		if line_type == LINE_TYPE_NOTE:
			note_type = int(split[3])
			# end of chart
			if note_type == NOTE_TYPE_END_OF_CHART:
				continue
			position = int(split[5])
			size = int(split[6])
			timestamp = conductor.get_timestamp(measure, tick)
			if notetype2channel[note_type] is None:
				continue
			if note_type in [NOTE_TYPE_HOLD_START, NOTE_TYPE_HOLD_START_R]:
				# hold starts are functionally equivalent to taps
				note = Note(position, size, note_type=NOTE_TYPE_TAP, timestamp=timestamp)
				notes.append(note)
				# but also record them for hold baking stuff
				hold = Hold()
				hold.add_note(Note(position, size, note_type=NOTE_TYPE_HOLD_SEGMENT, timestamp=timestamp))
				# the next hold segment shall come on lineno split[8]
				assert int(split[8]) > int(split[4])
				holds[int(split[8])] = hold
			elif note_type == NOTE_TYPE_HOLD_SEGMENT:
				hold = holds[int(split[4])]
				hold.add_note(Note(position, size, note_type=NOTE_TYPE_HOLD_SEGMENT, timestamp=timestamp))
				holds[int(split[8])] = hold
				del holds[int(split[4])]
			elif note_type == NOTE_TYPE_HOLD_END:
				hold = holds[int(split[4])]
				hold.add_note(Note(position, size, note_type=NOTE_TYPE_HOLD_SEGMENT, timestamp=timestamp))
				# add hold end note to note list for timing window checking
				note = Note(position, size, note_type=NOTE_TYPE_HOLD_END, timestamp=timestamp)
				notes.append(note)
			else:
				note = Note(position, size, note_type=note_type, timestamp=timestamp)
				notes.append(note)

	return notes, holds


def draw(notes: list, holds: dict):
	# chart setup
	min_timestamp = min([note.timestamp for note in notes])
	max_timestamp = max([note.timestamp for note in notes])
	chart_duration = max_timestamp - min_timestamp  # s
	resolution = DURATION_OF_1_FRAME / 2  # s/frame/2
	chart_length = chart_duration / resolution  # frames*2
	chart = np.zeros((21, int(np.round(chart_length) + 1), 180), dtype=np.int32)
	chart = np.pad(chart, ((0, 0), (20, 20), (0, 0)))  # pad for timing windows

	timestamp2index = lambda x: int(np.round(x / chart_duration * chart_length)) + 20
	index2timestamp = lambda x: (x - 20) / chart_length * chart_duration

	pos2index = lambda x: 3 * x + 2

	for note in notes:
		note.timestamp -= min_timestamp
		note.index = timestamp2index(note.timestamp)
	for hold in holds.values():
		for note in hold.notes:
			note.timestamp -= min_timestamp
			note.index = timestamp2index(note.timestamp)
		hold.bake_notes()

	# draw notes
	# note positions are tripled
	# (0|0|0) empty
	# (0|0|1) note starts here
	# (1|1|1) note passes through here (or all for size 60 notes)
	# (1|0|0) note ends here
	# (1|0|1) note starts and ends here (overlapped)
	def plot_note(note: Note):
		index = timestamp2index(note.timestamp) if note.index is None else note.index
		channel = notetype2channel[note.note_type] if note.channel is None else note.channel
		try:
			if note.size == 60:
				chart[channel, index] += 1
			else:
				# notes start on odd indices and end on even indices
				start_pos_index = pos2index(note.position)
				end_pos_index = start_pos_index + (note.size - 1) * 3 - 1
				if end_pos_index > 180:
					end_pos_index -= 180
					chart[channel, index, start_pos_index:] += 1
					chart[channel, index, :end_pos_index] += 1
				else:
					chart[channel, index, start_pos_index:end_pos_index] += 1
		except Exception as e:
			print(note)
			print(index)
			print(len(chart[0]))
			raise e

	def check_overlap(note1: Note, note2: Note):
		if note1.size == 60 or note2.size == 60:
			return True
		start1 = note1.position
		start2 = note2.position
		end1 = (start1 + note1.size - 1) % 60
		end2 = (start2 + note2.size - 1) % 60

		if start1 < end1 and start2 < end2:
			# notes don't wraparound
			return start1 < end2 and start2 < end1
		else:
			# some note wraps around
			return start1 < end2 or start2 < end1

	notes = np.array(notes, dtype="object")
	note_indices = np.array([note.index for note in notes])
	arange = np.arange(len(note_indices))
	for note in notes:
		if note.note_type != NOTE_TYPE_HOLD_END:
			plot_note(note)
		# plot timing windows
		if note.note_type in [NOTE_TYPE_HOLD_SEGMENT, NOTE_TYPE_HOLD_END] or note.channel in [CHANNEL_HOLD,
																							  CHANNEL_HOLD_OVERLAPPING]:
			# hold notes don't have timing windows aside from start
			continue
		else:
			assert note.note_type is not None
			timing_window = TIMING_WINDOWS[note.channel]

			## handle early window
			# check if on hold end
			same_index_mask = note_indices == note.index
			same_index_candidate_notes = notes[same_index_mask]
			overlapping_hold_ends = [candidate_note for candidate_note in same_index_candidate_notes if
									 check_overlap(note,
												   candidate_note) and candidate_note.note_type == NOTE_TYPE_HOLD_END]
			on_hold_end = len(overlapping_hold_ends) > 0
			early_window_size = timing_window[1 if on_hold_end else 0]
			early_index_mask = (note_indices >= note.index - early_window_size) & (note_indices < note.index)
			early_candidate_notes = notes[early_index_mask]
			early_overlapping_notes = [candidate_note for candidate_note in early_candidate_notes if
									   check_overlap(note, candidate_note)]
			if len(early_overlapping_notes) == 0:
				start_index = note.index - timing_window[0]
			else:
				start_index = int((note.index + early_overlapping_notes[-1].index) / 2)
			if on_hold_end:
				start_index = max(start_index, note.index - timing_window[1])

			# draw unsafe early window
			# if start_index > safe window border, this loop is skipped
			for index in np.arange(start_index, note.index - timing_window[1]):
				plot_note(Note(note.position, note.size, channel=timing_window[-1], index=index))

			# draw safe early window
			for index in np.arange(max(start_index, note.index - timing_window[1]), note.index):
				plot_note(Note(note.position, note.size, channel=timing_window[-2], index=index))

			## handle late window
			late_window_size = timing_window[3]
			late_index_mask = (note_indices <= note.index + late_window_size) & (note_indices > note.index)
			late_candidate_notes = notes[late_index_mask]
			late_overlapping_notes = [candidate_note for candidate_note in late_candidate_notes if
									  check_overlap(note, candidate_note)]
			if len(late_overlapping_notes) == 0:
				end_index = note.index + timing_window[3]
			else:
				end_index = int((note.index + late_overlapping_notes[0].index) / 2)

			# draw safe late window
			for index in np.arange(note.index, min(note.index + timing_window[2], end_index) + 1):
				plot_note(Note(note.position, note.size, channel=timing_window[-2], index=index))

			# draw unsafe late window
			for index in np.arange(note.index + timing_window[2] + 1, end_index + 1):
				plot_note(Note(note.position, note.size, channel=timing_window[-1], index=index))

	for hold in holds.values():
		prev_note = None
		for note in hold.notes:
			if prev_note is None:
				plot_note(note)
				prev_note = note
			else:
				# interpolate hold
				if prev_note.index == note.index:
					continue
				indices = np.arange(prev_note.index, note.index)
				if len(indices) < 2:
					# hold moves too fast, no need to interpolate this segment
					continue

				plot_note(note)
				# just assume it's baked idk
				interpolated_sizes = [prev_note.size for _ in indices]
				interpolated_positions = [prev_note.position for _ in indices]

				for i, (position, size, index) in enumerate(zip(interpolated_positions, interpolated_sizes, indices)):
					# skip first one, already drawn
					if i == 0:
						continue
					plot_note(Note(position, size, note_type=NOTE_TYPE_HOLD_SEGMENT, index=index))
				prev_note = note
	# overlapping taps/hold starts
	chart[CHANNEL_TAP_OVERLAPPING] = (chart[CHANNEL_TAP] > 1).astype(np.int32)
	chart[CHANNEL_TAP] = chart[CHANNEL_TAP].astype(bool).astype(np.int32)
	# overlapping holds
	chart[CHANNEL_HOLD_OVERLAPPING] = (chart[CHANNEL_HOLD] > 1).astype(np.int32)
	chart[CHANNEL_HOLD] = chart[CHANNEL_HOLD].astype(bool).astype(np.int32)

	for channel in SAFE_WINDOW_CHANNELS + UNSAFE_WINDOW_CHANNELS:
		chart[channel] = chart[channel].astype(bool).astype(np.int32)

	return chart


def visualize(chart, draw_windows=False):
	img = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	chart = chart.astype(bool).astype(np.uint8).reshape(chart.shape[0], chart.shape[1], chart.shape[2], 1)

	# holds
	img += chart[CHANNEL_HOLD] * 128
	img += chart[CHANNEL_HOLD_OVERLAPPING] * 64

	if draw_windows:
		# safe window
		temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
		for channel in SAFE_WINDOW_CHANNELS:
			temp += chart[channel]
		temp = np.clip(temp, 0, 1)
		mask = temp * 0.5
		mask[mask == 0] = 1
		img = img * mask
		temp[:, :, 0] *= 128
		temp[:, :, 1] *= 27
		temp[:, :, 2] *= 60
		img += temp

		# unsafe window
		temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
		for channel in UNSAFE_WINDOW_CHANNELS:
			temp += chart[channel]
		temp = np.clip(temp, 0, 1)
		mask = temp * 0.5
		mask[mask == 0] = 1
		img = img * mask
		temp[:, :, 0] *= 40
		temp[:, :, 1] *= 88
		temp[:, :, 2] *= 128
		img += temp
		img = img.astype(np.uint8)

	# guide lines
	img[:, 0] += 48
	img[:, 89] += 48

	# tap
	temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	temp += chart[CHANNEL_TAP]
	temp[:, :, 0] *= 255
	temp[:, :, 2] *= 255
	img *= (1 - chart[CHANNEL_TAP])
	img += temp

	# snap forward
	temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	temp += chart[CHANNEL_SNAP_FORWARD]
	temp[:, :, 0] *= 255
	img *= (1 - chart[CHANNEL_SNAP_FORWARD])
	img += temp

	# snap backward
	temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	temp += chart[CHANNEL_SNAP_BACKWARD]
	temp[:, :, 2] *= 255
	img *= (1 - chart[CHANNEL_SNAP_BACKWARD])
	img += temp

	# swipe CW (orange)
	temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	temp += chart[CHANNEL_SWIPE_CW]
	temp[:, :, 0] *= 255
	temp[:, :, 1] *= 170
	img *= (1 - chart[CHANNEL_SWIPE_CW])
	img += temp

	# swipe CCW (green)
	temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	temp += chart[CHANNEL_SWIPE_CCW]
	temp[:, :, 1] *= 255
	img *= (1 - chart[CHANNEL_SWIPE_CCW])
	img += temp

	# chain
	temp = np.zeros((chart.shape[1], chart.shape[2], 3), dtype=np.uint8)
	temp += chart[CHANNEL_CHAIN]
	temp[:, :, 0] *= 255
	temp[:, :, 1] *= 255
	img *= (1 - chart[CHANNEL_CHAIN])
	img += temp

	img = np.clip(img, 0, 255)

	COL_HEIGHT = 2400
	MARGIN_WIDTH = 50
	img = np.roll(img, -45, axis=-2)
	cols = int(len(temp) / COL_HEIGHT) + 1
	img = np.pad(img, ((0, COL_HEIGHT * cols - len(img)), (0, 0), (0, 0)))
	img = np.pad(img, ((0, 0), (1, 1), (0, 0)), constant_values=255)
	imgs = [np.pad(img[i:i + COL_HEIGHT], ((0, 0), (0, MARGIN_WIDTH), (0, 0))) for i in range(0, len(img), COL_HEIGHT)]
	img = np.concatenate(imgs, axis=1)[::-1]
	img = Image.fromarray(img)
	imgdraw = ImageDraw.Draw(img)
	max_time = int(chart.shape[1] / 120)
	for i in range(max_time):
		# imgdraw.text((183,2400-10), 'test', fill=(255,255,255))
		col = int(i / 20)
		row = i % 20
		imgdraw.text((184 + col * (182 + MARGIN_WIDTH), 2390 - row * 120), str(i) + '.0')
	return img


def button_clicked(*args, **kwargs):
	fileList = document.getElementById('fileInput').files
	for f in fileList:
		# reader is a pyodide.JsProxy
		reader = FileReader.new()
		# Create a Python proxy for the callback function
		onload_event = create_proxy(read_complete)
		# console.log("done")
		reader.onload = onload_event
		reader.readAsText(f)

def read_complete(event):
	# event is ProgressEvent
	content = document.getElementById("outputImage");
	img = visualize(draw(*parse_mer(event.target.result)), draw_windows=False)
	buffer = io.BytesIO()
	img.save(buffer, format='PNG')
	base64_img = base64.b64encode(buffer.getvalue()).decode()
	content.src = base64_img
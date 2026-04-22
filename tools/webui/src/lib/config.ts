// UI constants
export const PROPS_POLL_MS = 10000;
export const FETCH_TIMEOUT_MS = 2000;
export const JOB_POLL_MS = 2000;
export const SSE_RECONNECT_MS = 2000;
export const LOG_MAX_LINES = 50;
export const WAVEFORM_HEIGHT = 64;

// task types (mirrors task-types.h)
export const TASK_TEXT2MUSIC = 'text2music';
export const TASK_COVER = 'cover';
export const TASK_COVER_NOFSQ = 'cover-nofsq';
export const TASK_REPAINT = 'repaint';
export const TASK_LEGO = 'lego';
export const TASK_EXTRACT = 'extract';
export const TASK_COMPLETE = 'complete';

export const VALID_MP3_BITRATES = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];

export const TRACK_NAMES = [
	'vocals',
	'backing_vocals',
	'drums',
	'bass',
	'guitar',
	'keyboard',
	'percussion',
	'strings',
	'synth',
	'fx',
	'brass',
	'woodwinds'
] as const;

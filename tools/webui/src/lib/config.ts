// UI constants
export const PROPS_POLL_MS = 10000;
export const FETCH_TIMEOUT_MS = 2000;
export const JOB_POLL_MS = 2000;
export const SSE_RECONNECT_MS = 2000;
export const LOG_MAX_LINES = 50;
export const WAVEFORM_HEIGHT = 64;
export const WAVEFORM_BINS = 4096;

// task types (mirrors task-types.h)
export const TASK_TEXT2MUSIC = 'text2music';
export const TASK_COVER = 'cover';
export const TASK_COVER_NOFSQ = 'cover-nofsq';
export const TASK_REPAINT = 'repaint';
export const TASK_LEGO = 'lego';
export const TASK_EXTRACT = 'extract';
export const TASK_COMPLETE = 'complete';

// Solver names (mirrors task-types.h SOLVER_* and the C++ solver registry).
// The string is the canonical solver key resolved by solver_lookup().
export const SOLVER_EULER = 'euler';
export const SOLVER_SDE = 'sde';
export const SOLVER_DPM3M = 'dpm3m';
export const SOLVER_STORK4 = 'stork4';

// DCW modes (mirrors task-types.h DCW_MODE_*)
export const DCW_MODE_LOW = 'low';
export const DCW_MODE_HIGH = 'high';
export const DCW_MODE_DOUBLE = 'double';
export const DCW_MODE_PIX = 'pix';

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

import type { AceRequest, AceProps, Song } from './types.js';

const STORAGE_KEY = 'ace';

interface Saved {
	name: string;
	volume: number;
	format: string;
	mp3Bitrate: number;
	dark: boolean;
	logsOpen: boolean;
	request: AceRequest;
}

function load(): Saved {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const parsed = JSON.parse(raw);
			const validBitrates = [96, 128, 160, 192, 256, 320];
			return {
				name: parsed.name || '',
				volume: parsed.volume ?? 0.5,
				format: ['mp3', 'wav16', 'wav24', 'wav32'].includes(parsed.format) ? parsed.format : 'mp3',
				mp3Bitrate: validBitrates.includes(parsed.mp3Bitrate) ? parsed.mp3Bitrate : 0,
				dark: parsed.dark ?? true,
				logsOpen: parsed.logsOpen ?? true,
				request: parsed.request || { caption: '' }
			};
		}
	} catch {
		// corrupt or unavailable
	}
	return {
		name: '',
		volume: 0.5,
		format: 'mp3',
		mp3Bitrate: 0,
		dark: false,
		logsOpen: true,
		request: { caption: '' }
	};
}

const saved = load();

export const app = $state({
	name: saved.name,
	volume: saved.volume,
	format: saved.format,
	mp3Bitrate: saved.mp3Bitrate,
	dark: saved.dark,
	logsOpen: saved.logsOpen,
	request: saved.request as AceRequest,
	songs: [] as Song[],
	props: null as AceProps | null,
	toast: '' as string,
	toastOk: false,
	pendingRequests: [] as AceRequest[],
	pendingIndex: 0,
	refSongId: null as number | null,
	srcSongId: null as number | null,
	srcRangeStart: null as number | null,
	srcRangeEnd: null as number | null
});

let toastTimer = 0;

export function toast(msg: string, ms = 4000, ok = false) {
	clearTimeout(toastTimer);
	app.toast = msg;
	app.toastOk = ok;
	toastTimer = setTimeout(() => {
		app.toast = '';
	}, ms) as unknown as number;
}

// overwrite app.request, preserving model routing fields unless the
// incoming request provides them (non-empty string / non-null number).
export function setRequest(incoming: AceRequest) {
	if (!incoming.synth_model) incoming.synth_model = app.request.synth_model;
	if (!incoming.lm_model) incoming.lm_model = app.request.lm_model;
	if (!incoming.lora) incoming.lora = app.request.lora;
	if (incoming.lora_scale == null) incoming.lora_scale = app.request.lora_scale;
	app.request = incoming;
	app.srcRangeStart = incoming.repainting_start ?? null;
	app.srcRangeEnd = incoming.repainting_end ?? null;
}

// sync srcRange to request fields (srcRange is the UI source of truth,
// request fields are the serialization layer read by FIELDS helpers)
$effect.root(() => {
	$effect(() => {
		app.request.repainting_start = app.srcRangeStart ?? undefined;
		app.request.repainting_end = app.srcRangeEnd ?? undefined;
	});
});

// persist on every change
$effect.root(() => {
	$effect(() => {
		const data: Saved = {
			name: app.name,
			volume: app.volume,
			format: app.format,
			mp3Bitrate: app.mp3Bitrate,
			dark: app.dark,
			logsOpen: app.logsOpen,
			request: app.request
		};
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	});
});

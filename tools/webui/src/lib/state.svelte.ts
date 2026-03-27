import type { AceRequest, AceHealth, Song } from './types.js';

const STORAGE_KEY = 'ace';

interface Saved {
	name: string;
	volume: number;
	format: string;
	dark: boolean;
	logsOpen: boolean;
	request: AceRequest;
}

function load(): Saved {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const parsed = JSON.parse(raw);
			return {
				name: parsed.name || '',
				volume: parsed.volume ?? 0.5,
				format: parsed.format === 'wav' ? 'wav' : 'mp3',
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
	dark: saved.dark,
	logsOpen: saved.logsOpen,
	request: saved.request as AceRequest,
	songs: [] as Song[],
	health: null as AceHealth | null,
	toast: '' as string,
	pendingRequests: [] as AceRequest[],
	pendingIndex: 0,
	refSongId: null as number | null
});

let toastTimer = 0;

export function toast(msg: string, ms = 4000) {
	clearTimeout(toastTimer);
	app.toast = msg;
	toastTimer = setTimeout(() => {
		app.toast = '';
	}, ms) as unknown as number;
}

// persist on every change
$effect.root(() => {
	$effect(() => {
		const data: Saved = {
			name: app.name,
			volume: app.volume,
			format: app.format,
			dark: app.dark,
			logsOpen: app.logsOpen,
			request: app.request
		};
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	});
});

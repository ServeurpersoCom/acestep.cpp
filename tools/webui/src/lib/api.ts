import type { AceRequest, AceHealth } from './types.js';
import { FETCH_TIMEOUT_MS } from './config.js';

// POST lm: partial request -> enriched request(s)
export async function lmGenerate(req: AceRequest): Promise<AceRequest[]> {
	const res = await fetch('lm', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
	if (res.status === 503) throw new Error('Server busy');
	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: res.statusText }));
		throw new Error(err.error || res.statusText);
	}
	return res.json();
}

// POST synth[?wav=1]: enriched request -> audio blob + headers
export interface SynthResult {
	audio: Blob;
	seed: number;
	duration: number;
	computeMs: number;
}

export async function synthGenerate(req: AceRequest, format: string): Promise<SynthResult> {
	const url = format === 'wav' ? 'synth?wav=1' : 'synth';
	const res = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
	if (res.status === 503) throw new Error('Server busy');
	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: res.statusText }));
		throw new Error(err.error || res.statusText);
	}
	return {
		audio: await res.blob(),
		seed: Number(res.headers.get('X-Seed') || 0),
		duration: Number(res.headers.get('X-Duration') || 0),
		computeMs: Number(res.headers.get('X-Compute-Ms') || 0)
	};
}

// GET props: server config, pipeline status, default request (2s timeout)
export async function props(): Promise<AceHealth> {
	const res = await fetch('props', {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error('Server unreachable');
	return res.json();
}

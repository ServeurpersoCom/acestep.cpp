import type { AceRequest, AceProps } from './types.js';
import { FETCH_TIMEOUT_MS, JOB_POLL_MS } from './config.js';

// shared: submit a request and return the job ID
async function submitJob(url: string, init: RequestInit): Promise<string> {
	const res = await fetch(url, init);
	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: res.statusText }));
		throw new Error(`${res.status} ${err.error || res.statusText}`);
	}
	const data = await res.json();
	return data.id;
}

// POST /lm: submit LM request, returns job ID
export function lmSubmit(req: AceRequest): Promise<string> {
	return submitJob('lm', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
}

// POST /lm?mode=inspire: submit inspire request, returns job ID
export function lmSubmitInspire(req: AceRequest): Promise<string> {
	return submitJob('lm?mode=inspire', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
}

// POST /lm?mode=format: submit format request, returns job ID
export function lmSubmitFormat(req: AceRequest): Promise<string> {
	return submitJob('lm?mode=format', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(req)
	});
}

// POST /synth: submit synth request, returns job ID
export function synthSubmit(reqs: AceRequest[], format: string, mp3Bitrate = 0): Promise<string> {
	const url = format !== 'mp3' ? `synth?format=${format}` : 'synth';
	// Inject mp3_bitrate into the request object(s) when encoding as MP3 with a valid preset bitrate.
	// mp3_bitrate is a ServerFields key, not an AceRequest field — inject at serialization time only.
	const inject = format === 'mp3' && mp3Bitrate > 0 ? { mp3_bitrate: mp3Bitrate } : {};
	const single = { ...reqs[0], ...inject };
	const body =
		reqs.length === 1
			? JSON.stringify(single)
			: JSON.stringify(reqs.map((r) => ({ ...r, ...inject })));
	return submitJob(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body
	});
}

// POST /synth (multipart): submit synth request with audio files, returns job ID
export function synthSubmitWithAudio(
	reqs: AceRequest[],
	srcAudio: Blob | null,
	refAudio: Blob | null,
	format: string,
	mp3Bitrate = 0
): Promise<string> {
	const url = format !== 'mp3' ? `synth?format=${format}` : 'synth';
	const inject = format === 'mp3' && mp3Bitrate > 0 ? { mp3_bitrate: mp3Bitrate } : {};
	const single = { ...reqs[0], ...inject };
	const body =
		reqs.length === 1
			? JSON.stringify(single)
			: JSON.stringify(reqs.map((r) => ({ ...r, ...inject })));
	const form = new FormData();
	form.append('request', new Blob([body], { type: 'application/json' }), 'request.json');
	if (srcAudio) form.append('audio', srcAudio, 'src.audio');
	if (refAudio) form.append('ref_audio', refAudio, 'ref.audio');
	return submitJob(url, { method: 'POST', body: form });
}

// GET /job?id=X: poll job status
export async function jobStatus(id: string): Promise<string> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}`, {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error(`${res.status} Job not found`);
	const data = await res.json();
	return data.status;
}

// poll until done, throws on failure or cancel.
// no timeout: long jobs (XL synth) can take 10+ minutes.
// the user cancels via the Cancel button if needed.
// retries on network errors (TypeError) and timeouts (DOMException).
// propagates HTTP errors (404 = job evicted, server restarted).
export async function pollJob(id: string): Promise<void> {
	for (;;) {
		try {
			const status = await jobStatus(id);
			if (status === 'done') return;
			if (status === 'failed') throw new Error('Generation failed');
			if (status === 'cancelled') throw new Error('Cancelled');
		} catch (e) {
			if (e instanceof TypeError || e instanceof DOMException) {
				// network down or timeout: retry next cycle
			} else {
				throw e;
			}
		}
		await new Promise((r) => setTimeout(r, JOB_POLL_MS));
	}
}

// GET /job?id=X&result=1: fetch result as JSON array (for LM jobs)
export async function jobResultJson(id: string): Promise<AceRequest[]> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`);
	if (!res.ok) throw new Error(`${res.status} Result not ready`);
	return res.json();
}

// GET /job?id=X&result=1: fetch result as audio blobs (for synth jobs)
export async function jobResultBlobs(id: string): Promise<Blob[]> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`);
	if (!res.ok) throw new Error(`${res.status} Result not ready`);
	const ct = res.headers.get('Content-Type') || '';
	if (!ct.startsWith('multipart/')) {
		return [await res.blob()];
	}
	const match = ct.match(/boundary=([^\s;]+)/);
	if (!match) throw new Error('Missing boundary in multipart response');
	const mime = ct.includes('wav') ? 'audio/wav' : 'audio/mpeg';
	return parseMultipart(new Uint8Array(await res.arrayBuffer()), match[1], mime);
}

// POST /job?id=X&cancel=1: cancel a specific job
export async function cancelJob(id: string): Promise<void> {
	await fetch(`job?id=${encodeURIComponent(id)}&cancel=1`, { method: 'POST' });
}

// parse multipart/mixed binary response into Blob[].
// each part has only Content-Type header + raw audio body.
function parseMultipart(buf: Uint8Array, boundary: string, mime: string): Blob[] {
	const enc = new TextEncoder();
	const delim = enc.encode('--' + boundary);
	const results: Blob[] = [];

	// find all boundary positions
	const positions: number[] = [];
	for (let i = 0; i <= buf.length - delim.length; i++) {
		let ok = true;
		for (let j = 0; j < delim.length; j++) {
			if (buf[i + j] !== delim[j]) {
				ok = false;
				break;
			}
		}
		if (ok) positions.push(i);
	}

	for (let p = 0; p < positions.length - 1; p++) {
		const partStart = positions[p] + delim.length + 2;
		const partEnd = positions[p + 1] - 2;
		if (partStart >= partEnd) continue;

		// split headers from body at \r\n\r\n
		let splitAt = -1;
		for (let i = partStart; i < partEnd - 3; i++) {
			if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) {
				splitAt = i;
				break;
			}
		}
		if (splitAt < 0) continue;

		const body = buf.slice(splitAt + 4, partEnd);
		results.push(new Blob([body], { type: mime }));
	}

	return results;
}

// POST /understand (multipart): submit understand request, returns job ID
export function understandSubmit(
	blob: Blob,
	lmModel?: string,
	synthModel?: string
): Promise<string> {
	const form = new FormData();
	form.append('audio', blob, 'input.audio');
	const fields: Record<string, string> = {};
	if (lmModel) fields.lm_model = lmModel;
	if (synthModel) fields.synth_model = synthModel;
	if (Object.keys(fields).length > 0) {
		form.append(
			'request',
			new Blob([JSON.stringify(fields)], { type: 'application/json' }),
			'request.json'
		);
	}
	return submitJob('understand', { method: 'POST', body: form });
}

// GET /props: server config (2s timeout)
export async function props(): Promise<AceProps> {
	const res = await fetch('props', {
		signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
	});
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	return res.json();
}

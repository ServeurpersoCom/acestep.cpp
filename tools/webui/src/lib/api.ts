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

// POST /lm with lm_mode="inspire": returns job ID
export function lmSubmitInspire(req: AceRequest): Promise<string> {
	return submitJob('lm', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ ...req, lm_mode: 'inspire' })
	});
}

// POST /lm with lm_mode="format": returns job ID
export function lmSubmitFormat(req: AceRequest): Promise<string> {
	return submitJob('lm', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ ...req, lm_mode: 'format' })
	});
}

// POST /synth: submit synth request, returns job ID
export function synthSubmit(reqs: AceRequest[], format: string): Promise<string> {
	const reqsWithFormat = reqs.map((r) => ({ ...r, output_format: format }));
	const body =
		reqsWithFormat.length === 1
			? JSON.stringify(reqsWithFormat[0])
			: JSON.stringify(reqsWithFormat);
	return submitJob('synth', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body
	});
}

// POST /synth (multipart): submit synth request with audio files and/or latents,
// returns job ID. When latents are provided for a side, they win over the matching
// audio: the server skips that VAE encode entirely.
export function synthSubmitWithAudio(
	reqs: AceRequest[],
	srcAudio: Blob | null,
	srcLatents: Blob | null,
	refAudio: Blob | null,
	refLatents: Blob | null,
	format: string
): Promise<string> {
	const reqsWithFormat = reqs.map((r) => ({ ...r, output_format: format }));
	const body =
		reqsWithFormat.length === 1
			? JSON.stringify(reqsWithFormat[0])
			: JSON.stringify(reqsWithFormat);
	const form = new FormData();
	form.append('request', new Blob([body], { type: 'application/json' }), 'request.json');
	if (srcLatents) form.append('src_latents', srcLatents, 'src.latents');
	else if (srcAudio) form.append('audio', srcAudio, 'src.audio');
	if (refLatents) form.append('ref_latents', refLatents, 'ref.latents');
	else if (refAudio) form.append('ref_audio', refAudio, 'ref.audio');
	return submitJob('synth', { method: 'POST', body: form });
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

// Synth/vae-decode result. The server always replies multipart/mixed for
// /synth (one audio part per generated track plus the cover latents when
// produced) and raw audio for /vae decode path (single Content-Type, no
// latents in the body since the client already holds the ones it uploaded).
export interface SynthResult {
	audios: Blob[];
	latents: Blob | null;
}

// GET /job?id=X&result=1: fetch synth result. Multipart parts are discriminated
// by their own Content-Type header: audio/* parts populate audios, the
// application/octet-stream part is the cover latents. A single-Content-Type
// response (/vae decode path) is returned as one audio part with no latents.
export async function jobResultBlobs(id: string): Promise<SynthResult> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`);
	if (!res.ok) throw new Error(`${res.status} Result not ready`);
	const ct = res.headers.get('Content-Type') || '';
	if (!ct.startsWith('multipart/')) {
		return { audios: [await res.blob()], latents: null };
	}
	const match = ct.match(/boundary=([^\s;]+)/);
	if (!match) throw new Error('Missing boundary in multipart response');
	return parseMultipartTyped(new Uint8Array(await res.arrayBuffer()), match[1]);
}

// GET /job?id=X&result=1: fetch understand result. The server replies
// multipart/mixed with one application/json part (the AceRequest array) and
// the cover latents. The latents are exposed via the second return so callers
// can attach them to the originating Song.
export interface UnderstandResult {
	requests: AceRequest[];
	latents: Blob | null;
}

export async function jobResultUnderstand(id: string): Promise<UnderstandResult> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`);
	if (!res.ok) throw new Error(`${res.status} Result not ready`);
	const ct = res.headers.get('Content-Type') || '';
	if (!ct.startsWith('multipart/')) {
		// Single-part fallback: pure JSON, no latents.
		return { requests: await res.json(), latents: null };
	}
	const match = ct.match(/boundary=([^\s;]+)/);
	if (!match) throw new Error('Missing boundary in multipart response');
	const parts = parseMultipartParts(new Uint8Array(await res.arrayBuffer()), match[1]);
	let requests: AceRequest[] = [];
	let latents: Blob | null = null;
	for (const part of parts) {
		if (part.contentType.startsWith('application/json')) {
			requests = JSON.parse(await part.body.text());
		} else if (part.contentType.startsWith('application/octet-stream')) {
			latents = part.body;
		}
	}
	return { requests, latents };
}

// POST /job?id=X&cancel=1: cancel a specific job
export async function cancelJob(id: string): Promise<void> {
	await fetch(`job?id=${encodeURIComponent(id)}&cancel=1`, { method: 'POST' });
}

// Raw part of a multipart/mixed response. Each part carries its own
// Content-Type, the higher-level parsers (jobResultBlobs, jobResultUnderstand)
// dispatch on that header to assemble typed results.
interface MultipartPart {
	contentType: string;
	body: Blob;
}

// Parse a multipart/mixed binary response into typed parts. Reads only the
// Content-Type header on each part; other headers (Content-Disposition...)
// are ignored. Returns parts in wire order.
function parseMultipartParts(buf: Uint8Array, boundary: string): MultipartPart[] {
	const enc = new TextEncoder();
	const delim = enc.encode('--' + boundary);
	const dec = new TextDecoder();
	const results: MultipartPart[] = [];

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

		// scan headers for Content-Type. Headers are CRLF-separated ASCII.
		const headerText = dec.decode(buf.slice(partStart, splitAt));
		let contentType = 'application/octet-stream';
		for (const line of headerText.split(/\r\n/)) {
			const m = line.match(/^Content-Type:\s*(.+)$/i);
			if (m) {
				contentType = m[1].trim();
				break;
			}
		}

		const body = buf.slice(splitAt + 4, partEnd);
		results.push({ contentType, body: new Blob([body], { type: contentType }) });
	}

	return results;
}

// Split typed parts into audio blobs and an optional latents blob. Audio parts
// keep their per-part mime (audio/mpeg or audio/wav), the latents are the single
// application/octet-stream part if present.
function parseMultipartTyped(buf: Uint8Array, boundary: string): SynthResult {
	const parts = parseMultipartParts(buf, boundary);
	const audios: Blob[] = [];
	let latents: Blob | null = null;
	for (const part of parts) {
		if (part.contentType.startsWith('audio/')) {
			audios.push(part.body);
		} else if (part.contentType.startsWith('application/octet-stream')) {
			latents = part.body;
		}
	}
	return { audios, latents };
}

// POST /vae (multipart): single VAE entrypoint. Direction depends on which
// part you send: 'audio' goes encode (latents out), 'src_latents' goes
// decode (audio out). Mutually exclusive, server rejects both or neither.
// The result body carries only the opposite side: the client already holds
// the one it just uploaded.
export function vaeEncode(audio: Blob): Promise<string> {
	const form = new FormData();
	form.append('audio', audio, 'src.audio');
	return submitJob('vae', { method: 'POST', body: form });
}

export function vaeDecode(srcLatents: Blob, format: string): Promise<string> {
	const form = new FormData();
	form.append('src_latents', srcLatents, 'src.vae');
	if (format) {
		const body = JSON.stringify({ output_format: format });
		form.append('request', new Blob([body], { type: 'application/json' }), 'request.json');
	}
	return submitJob('vae', { method: 'POST', body: form });
}

// GET /job?id=X&result=1 for /vae encode path: raw .vae bytes (application/
// octet-stream, f32 [T*64] time-major, no header). Caller wraps the Blob
// into a Song.latents.
export async function jobResultLatents(id: string): Promise<Blob> {
	const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`);
	if (!res.ok) throw new Error(`${res.status} Result not ready`);
	return res.blob();
}

// POST /understand (multipart): submit understand request, returns job ID.
// Source can be raw audio or pre-encoded latents; latents win when both are
// provided, skipping the VAE encode on the server side.
export function understandSubmit(
	audio: Blob | null,
	srcLatents: Blob | null,
	lmModel?: string,
	synthModel?: string
): Promise<string> {
	const form = new FormData();
	if (srcLatents) form.append('src_latents', srcLatents, 'src.latents');
	else if (audio) form.append('audio', audio, 'input.audio');
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

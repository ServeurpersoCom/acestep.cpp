// field descriptor table: single source of truth for AceRequest field knowledge.
//
// every field in AceRequest is listed here with its section and type.
// helpers derive clear, preserve, serialize, and pick logic from this table.
// adding a field = adding one line here.

import type { AceRequest } from './types.js';

export type FieldSection =
	| 'content'
	| 'metadata'
	| 'lm'
	| 'flow'
	| 'advanced'
	| 'toolbar'
	| 'routing';

interface FieldDef {
	key: keyof AceRequest;
	section: FieldSection;
	type: 'str' | 'num' | 'bool';
	min?: number;
}

export const FIELDS: readonly FieldDef[] = [
	// content: per-variation, replaced by LM results
	{ key: 'caption', section: 'content', type: 'str' },
	{ key: 'lyrics', section: 'content', type: 'str' },
	{ key: 'audio_codes', section: 'content', type: 'str' },
	{ key: 'use_cot_caption', section: 'content', type: 'bool' },

	// metadata: per-variation, cleared by Clear metadata
	{ key: 'vocal_language', section: 'metadata', type: 'str' },
	{ key: 'bpm', section: 'metadata', type: 'num' },
	{ key: 'duration', section: 'metadata', type: 'num' },
	{ key: 'keyscale', section: 'metadata', type: 'str' },
	{ key: 'timesignature', section: 'metadata', type: 'str' },

	// LM control: user settings, preserved across Compose/variation switch
	{ key: 'lm_batch_size', section: 'lm', type: 'num', min: 1 },
	{ key: 'lm_temperature', section: 'lm', type: 'num' },
	{ key: 'lm_cfg_scale', section: 'lm', type: 'num' },
	{ key: 'lm_top_p', section: 'lm', type: 'num' },
	{ key: 'lm_top_k', section: 'lm', type: 'num' },
	{ key: 'lm_negative_prompt', section: 'lm', type: 'str' },
	{ key: 'lm_seed', section: 'metadata', type: 'num' },

	// flow matching: cleared by Clear flow matching, preserved across Compose
	{ key: 'inference_steps', section: 'flow', type: 'num' },
	{ key: 'guidance_scale', section: 'flow', type: 'num' },
	{ key: 'shift', section: 'flow', type: 'num' },
	{ key: 'audio_cover_strength', section: 'flow', type: 'num' },
	{ key: 'cover_noise_strength', section: 'flow', type: 'num' },
	{ key: 'repainting_start', section: 'flow', type: 'num' },
	{ key: 'repainting_end', section: 'flow', type: 'num' },
	{ key: 'seed', section: 'flow', type: 'num' },

	// advanced and post-processing: cleared by its own Clear, preserved across Compose
	{ key: 'custom_timesteps', section: 'advanced', type: 'str' },
	{ key: 'dcw_mode', section: 'advanced', type: 'str' },
	{ key: 'dcw_scaler', section: 'advanced', type: 'num' },
	{ key: 'dcw_high_scaler', section: 'advanced', type: 'num' },
	{ key: 'infer_method', section: 'advanced', type: 'str' },
	{ key: 'latent_shift', section: 'advanced', type: 'num' },
	{ key: 'latent_rescale', section: 'advanced', type: 'num' },
	{ key: 'peak_clip', section: 'advanced', type: 'num' },
	{ key: 'mp3_bitrate', section: 'advanced', type: 'num', min: 1 },

	// toolbar: preserved, not in any clear section
	{ key: 'synth_batch_size', section: 'toolbar', type: 'num', min: 1 },

	// routing: model and task selection, preserved across Compose
	{ key: 'task_type', section: 'routing', type: 'str' },
	{ key: 'track', section: 'routing', type: 'str' },
	{ key: 'synth_model', section: 'routing', type: 'str' },
	{ key: 'lm_model', section: 'routing', type: 'str' },
	{ key: 'adapter', section: 'routing', type: 'str' },
	{ key: 'adapter_scale', section: 'routing', type: 'num' },
	{ key: 'vae', section: 'routing', type: 'str' }
];

// convert to number, undefined if empty/NaN
export function num(v: unknown): number | undefined {
	if (v == null || v === '') return undefined;
	const n = Number(v);
	return isNaN(n) ? undefined : n;
}

// typed dynamic access helpers
type Rec = Record<string, unknown>;
function get(r: AceRequest | Partial<AceRequest>, key: keyof AceRequest): unknown {
	return (r as Rec)[key];
}
function set(r: AceRequest | Partial<AceRequest>, key: keyof AceRequest, val: unknown): void {
	(r as Rec)[key] = val;
}

// resolve a field value to its serialized form, undefined if empty
function resolveField(f: FieldDef, raw: unknown): unknown {
	if (f.type === 'num') {
		const n = num(raw);
		return n != null && (f.min == null || n >= f.min) ? n : undefined;
	}
	if (f.type === 'str') return raw ? String(raw) : undefined;
	return raw ?? undefined;
}

// serialize non-empty fields for JSON export.
// caller injects srcRange (repainting_start/end) and validates adapter.
export function buildSparse(r: AceRequest): AceRequest {
	const out: AceRequest = { caption: String(r.caption || '') };
	for (const f of FIELDS) {
		if (f.key === 'caption') continue;
		const val = resolveField(f, get(r, f.key));
		if (val !== undefined) set(out, f.key, val);
	}
	return out;
}

// reset all fields in a section to empty/undefined.
// caller handles srcRange for 'flow'.
export function clearSection(r: AceRequest, section: FieldSection): void {
	for (const f of FIELDS) {
		if (f.section !== section) continue;
		set(r, f.key, f.type === 'str' ? '' : undefined);
	}
}

// merge content from incoming with settings from current.
// content and metadata come from incoming, everything else from current.
export function withCurrentSettings(incoming: AceRequest, current: AceRequest): AceRequest {
	const out = { ...incoming };
	for (const f of FIELDS) {
		if (f.section === 'content' || f.section === 'metadata') continue;
		set(out, f.key, get(current, f.key));
	}
	return out;
}

// extract non-null fields from specified sections.
// caller removes per-expansion fields (seed, synth_batch_size) and injects srcRange.
export function pickSections(r: AceRequest, sections: FieldSection[]): Partial<AceRequest> {
	const out: Partial<AceRequest> = {};
	const allow = new Set(sections);
	for (const f of FIELDS) {
		if (!allow.has(f.section)) continue;
		const val = resolveField(f, get(r, f.key));
		if (val !== undefined) set(out, f.key, val);
	}
	return out;
}

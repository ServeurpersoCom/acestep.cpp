<script lang="ts">
	import { RotateCcw, Download, FolderOpen } from '@lucide/svelte';
	import { app } from '../lib/state.svelte.js';
	import { lmGenerate, synthGenerate } from '../lib/api.js';
	import { putSong } from '../lib/db.js';
	import type { AceRequest, Song } from '../lib/types.js';

	let busy = $state(false);
	let error = $state('');
	let fileInput: HTMLInputElement;

	let d = $derived(app.health?.default);

	function reset() {
		app.name = '';
		app.request = { caption: '' };
	}

	function exportJson() {
		const json = JSON.stringify(buildRequest(), null, 2);
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		const safe = app.name.replace(/[^a-zA-Z0-9 _-]/g, '') || 'request';
		a.download = `${safe}.json`;
		a.click();
		URL.revokeObjectURL(url);
	}

	function importJson() {
		fileInput.click();
	}

	function onFileSelected(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		file
			.text()
			.then((text) => {
				app.request = JSON.parse(text) as AceRequest;
			})
			.catch(() => {
				error = 'Invalid JSON file';
			});
	}

	// convert string or number to number, return undefined if empty/NaN
	function num(v: unknown): number | undefined {
		if (v == null || v === '') return undefined;
		const n = Number(v);
		return isNaN(n) ? undefined : n;
	}

	// snapshot app.request into a clean AceRequest with proper types.
	// bind:value guarantees app.request always matches the DOM.
	function buildRequest(): AceRequest {
		const r = app.request;
		const out: AceRequest = { caption: String(r.caption || '') };
		if (r.lyrics) out.lyrics = String(r.lyrics);
		if (r.audio_codes) out.audio_codes = String(r.audio_codes);
		if (r.vocal_language) out.vocal_language = String(r.vocal_language);
		if (r.keyscale) out.keyscale = String(r.keyscale);
		if (r.timesignature) out.timesignature = String(r.timesignature);
		const bpm = num(r.bpm);
		if (bpm != null) out.bpm = bpm;
		const duration = num(r.duration);
		if (duration != null) out.duration = duration;
		const seed = num(r.seed);
		if (seed != null) out.seed = seed;
		const lm_temperature = num(r.lm_temperature);
		if (lm_temperature != null) out.lm_temperature = lm_temperature;
		const lm_cfg_scale = num(r.lm_cfg_scale);
		if (lm_cfg_scale != null) out.lm_cfg_scale = lm_cfg_scale;
		const lm_top_p = num(r.lm_top_p);
		if (lm_top_p != null) out.lm_top_p = lm_top_p;
		const lm_top_k = num(r.lm_top_k);
		if (lm_top_k != null) out.lm_top_k = lm_top_k;
		const inference_steps = num(r.inference_steps);
		if (inference_steps != null) out.inference_steps = inference_steps;
		const guidance_scale = num(r.guidance_scale);
		if (guidance_scale != null) out.guidance_scale = guidance_scale;
		const shift = num(r.shift);
		if (shift != null) out.shift = shift;
		return out;
	}

	// Compose: send form to LM, fill ONLY empty fields with the response.
	// a non-empty field is never overwritten. the user is the sole authority.
	// audio_codes always comes from LM (we cleared it before sending).
	async function compose() {
		busy = true;
		error = '';
		try {
			const req = buildRequest();
			req.audio_codes = '';
			const results = await lmGenerate(req);
			if (results.length > 0) {
				const r = results[0];
				app.request.audio_codes = r.audio_codes || '';
				if (!app.request.lyrics && r.lyrics) app.request.lyrics = r.lyrics;
				if (!app.request.vocal_language && r.vocal_language)
					app.request.vocal_language = r.vocal_language;
				if (!app.request.keyscale && r.keyscale) app.request.keyscale = r.keyscale;
				if (!app.request.timesignature && r.timesignature)
					app.request.timesignature = r.timesignature;
				if (!app.request.bpm && r.bpm) app.request.bpm = r.bpm;
				if (!app.request.duration && r.duration) app.request.duration = r.duration;
				if (!app.request.seed && r.seed) app.request.seed = r.seed;
				if (!app.request.caption && r.caption) app.request.caption = r.caption;
			}
		} catch (e: unknown) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	// POST /synth: read form, send with format, store audio
	async function synthesize() {
		busy = true;
		error = '';
		try {
			const req = buildRequest();
			const result = await synthGenerate(req, app.format);
			const song = {
				name: app.name || 'Untitled',
				format: app.format,
				created: Date.now(),
				caption: req.caption,
				seed: result.seed,
				duration: result.duration,
				computeMs: result.computeMs,
				request: req,
				audio: result.audio
			} as Song;
			song.id = await putSong(song);
			app.songs.unshift(song);
		} catch (e: unknown) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	function ph(v: unknown): string {
		return v != null ? String(v) : '';
	}
</script>

<form class="request-form" onsubmit={(e) => e.preventDefault()}>
	<input type="file" accept=".json" bind:this={fileInput} onchange={onFileSelected} hidden />
	<div class="toolbar">
		<button type="button" onclick={importJson} title="Open"><FolderOpen size={14} /> Open</button>
		<button type="button" onclick={exportJson} title="Save"><Download size={14} /> Save</button>
		<button type="button" onclick={reset} title="Reset"><RotateCcw size={14} /> Reset</button>
	</div>

	<label
		>Name
		<input type="text" bind:value={app.name} placeholder="Untitled" />
	</label>

	<label
		>Caption
		<textarea
			rows="8"
			placeholder="Upbeat pop rock with driving guitars"
			bind:value={app.request.caption}
		></textarea>
	</label>

	<label
		>Lyrics
		<textarea
			rows="8"
			placeholder="[Instrumental] or write lyrics..."
			bind:value={app.request.lyrics}
		></textarea>
	</label>

	<details>
		<summary>Metadata</summary>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>Language <input
						type="text"
						placeholder={ph(d?.vocal_language)}
						bind:value={app.request.vocal_language}
					/></label
				>
				<label
					>BPM <input type="text" placeholder={ph(d?.bpm)} bind:value={app.request.bpm} /></label
				>
				<label
					>Duration <input
						type="text"
						placeholder={ph(d?.duration)}
						bind:value={app.request.duration}
					/></label
				>
				<label
					>Key <input
						type="text"
						placeholder={ph(d?.keyscale)}
						bind:value={app.request.keyscale}
					/></label
				>
				<label
					>Time sig <input
						type="text"
						placeholder={ph(d?.timesignature)}
						bind:value={app.request.timesignature}
					/></label
				>
				<label
					>Seed <input type="text" placeholder={ph(d?.seed)} bind:value={app.request.seed} /></label
				>
			</div>
		</div>
	</details>

	<details>
		<summary>Advanced LM</summary>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>Temperature <input
						type="text"
						placeholder={ph(d?.lm_temperature)}
						bind:value={app.request.lm_temperature}
					/></label
				>
				<label
					>CFG scale <input
						type="text"
						placeholder={ph(d?.lm_cfg_scale)}
						bind:value={app.request.lm_cfg_scale}
					/></label
				>
				<label
					>Top P <input
						type="text"
						placeholder={ph(d?.lm_top_p)}
						bind:value={app.request.lm_top_p}
					/></label
				>
				<label
					>Top K <input
						type="text"
						placeholder={ph(d?.lm_top_k)}
						bind:value={app.request.lm_top_k}
					/></label
				>
			</div>
		</div>
	</details>

	<details>
		<summary>Audio codes</summary>
		<div class="details-body">
			<label>
				<textarea
					rows="8"
					placeholder="Filled by Compose, or paste for dit-only"
					bind:value={app.request.audio_codes}
				></textarea>
			</label>
		</div>
	</details>

	<button type="button" disabled={busy} onclick={compose}>Compose</button>

	<hr />

	<div class="format-pick">
		<label class="format-label">
			<input type="radio" name="format" value="mp3" bind:group={app.format} /> MP3
		</label>
		<label class="format-label">
			<input type="radio" name="format" value="wav" bind:group={app.format} /> WAV
		</label>
	</div>

	<details>
		<summary>Advanced Synth</summary>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>Steps <input
						type="text"
						placeholder={ph(d?.inference_steps)}
						bind:value={app.request.inference_steps}
					/></label
				>
				<label
					>CFG scale <input
						type="text"
						placeholder={ph(d?.guidance_scale)}
						bind:value={app.request.guidance_scale}
					/></label
				>
				<label
					>Shift <input
						type="text"
						placeholder={ph(d?.shift)}
						bind:value={app.request.shift}
					/></label
				>
			</div>
		</div>
	</details>

	<button type="button" disabled={busy} onclick={synthesize}>Synthesize</button>

	{#if error}
		<div class="error">{error}</div>
	{/if}
</form>

<style>
	.request-form {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.toolbar {
		display: flex;
		gap: 0.5rem;
	}
	.toolbar button {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.3rem;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.85rem;
		color: var(--fg-dim);
	}
	textarea,
	input[type='text'] {
		font-family: inherit;
		font-size: 0.9rem;
		padding: 0.4rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-input);
		color: var(--fg);
		resize: vertical;
	}
	textarea:focus,
	input:focus {
		outline: 2px solid var(--focus);
		outline-offset: -1px;
	}
	.meta-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(8rem, 1fr));
		gap: 0.5rem;
	}
	details summary {
		cursor: pointer;
		font-size: 0.85rem;
		color: var(--fg-dim);
		padding: 0.4rem 0;
	}
	details summary:hover {
		color: var(--fg);
	}
	.details-body {
		padding: 0.25rem 0 0.5rem;
	}
	hr {
		border: none;
		border-top: 1px solid var(--border);
	}
	.format-pick {
		display: flex;
		gap: 1rem;
	}
	.format-label {
		flex-direction: row;
		align-items: center;
		gap: 0.3rem;
		cursor: pointer;
	}
	.error {
		color: var(--error);
		font-size: 0.85rem;
	}
	button {
		padding: 0.5rem 1rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-btn);
		color: var(--fg);
		cursor: pointer;
		font-size: 0.85rem;
	}
	button:hover:not(:disabled) {
		background: var(--bg-btn-hover);
	}
	button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
</style>

<script lang="ts">
	import { RotateCcw, Download, FolderOpen } from '@lucide/svelte';
	import { app, toast, setRequest } from '../lib/state.svelte.js';
	import {
		lmGenerate,
		lmInspire,
		lmFormat,
		synthGenerate,
		synthGenerateWithAudio,
		understandAudio
	} from '../lib/api.js';
	import { putSong } from '../lib/db.js';
	import {
		TASK_COVER,
		TASK_REPAINT,
		TASK_LEGO,
		TASK_EXTRACT,
		TASK_COMPLETE,
		TRACK_NAMES
	} from '../lib/config.js';
	import type { AceRequest, Song } from '../lib/types.js';

	let busy = $state(false);
	let fileInput: HTMLInputElement;

	let d = $derived(app.props?.default);
	let ditModels = $derived(app.props?.models.dit ?? []);
	let lmModels = $derived(app.props?.models.lm ?? []);
	let loraList = $derived(app.props?.loras ?? []);
	let loraStale = $derived(!!app.request.lora && !loraList.includes(String(app.request.lora)));
	let taskType = $derived(app.request.task_type || '');
	let needsTrack = $derived(
		taskType === TASK_LEGO || taskType === TASK_EXTRACT || taskType === TASK_COMPLETE
	);

	// auto-fill track when switching to a mode that needs one, clear when leaving
	$effect(() => {
		if (needsTrack && !app.request.track) {
			app.request.track = TRACK_NAMES[0];
		} else if (!needsTrack && app.request.track) {
			app.request.track = '';
		}
	});

	function reset() {
		app.name = '';
		setRequest({ caption: '' });
		app.pendingRequests = [];
		app.pendingIndex = 0;
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
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		// reset so the same file can be re-opened
		input.value = '';

		const ext = file.name.split('.').pop()?.toLowerCase() || '';

		// JSON: load request into form (existing behavior)
		if (ext === 'json') {
			file
				.text()
				.then((text) => {
					setRequest(JSON.parse(text) as AceRequest);
					app.name = file.name.replace(/\.json$/i, '') || 'Imported';
					app.pendingRequests = [];
					app.pendingIndex = 0;
				})
				.catch(() => {
					toast('Invalid JSON file');
				});
			return;
		}

		// MP3 or WAV: send to /understand, populate form + create song card
		if (ext === 'mp3' || ext === 'wav') {
			importAudio(file, ext);
			return;
		}

		toast('Unsupported file type: ' + ext);
	}

	// import audio file via /understand endpoint.
	// creates a song card with the original audio and fills the form
	// with the returned metadata so it matches existing generated songs.
	async function importAudio(file: File, ext: string) {
		busy = true;
		try {
			toast('Understanding audio...', 4000, true);
			const blob = new Blob([await file.arrayBuffer()], {
				type: ext === 'wav' ? 'audio/wav' : 'audio/mpeg'
			});
			const result = await understandAudio(blob);

			setRequest(result);
			app.pendingRequests = [];
			app.pendingIndex = 0;

			// derive a clean name from the filename (strip extension)
			const name = file.name.replace(/\.(mp3|wav)$/i, '') || 'Imported';
			app.name = name;

			// create a song card so the audio is playable immediately
			const song: Song = {
				name: name,
				format: ext,
				created: Date.now(),
				caption: result.caption || '',
				seed: Number(result.seed) || 0,
				duration: Number(result.duration) || 0,
				request: { ...result },
				audio: blob
			};
			song.id = await putSong(song);
			app.songs.unshift(song);

			toast('Imported: ' + name, 4000, true);
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			busy = false;
		}
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
		if (r.lm_negative_prompt) out.lm_negative_prompt = String(r.lm_negative_prompt);
		const inference_steps = num(r.inference_steps);
		if (inference_steps != null) out.inference_steps = inference_steps;
		const guidance_scale = num(r.guidance_scale);
		if (guidance_scale != null) out.guidance_scale = guidance_scale;
		const shift = num(r.shift);
		if (shift != null) out.shift = shift;
		const audio_cover_strength = num(r.audio_cover_strength);
		if (audio_cover_strength != null) out.audio_cover_strength = audio_cover_strength;
		const cover_noise_strength = num(r.cover_noise_strength);
		if (cover_noise_strength != null) out.cover_noise_strength = cover_noise_strength;
		const lm_batch_size = num(r.lm_batch_size);
		if (lm_batch_size != null && lm_batch_size >= 1) out.lm_batch_size = lm_batch_size;
		const synth_batch_size = num(r.synth_batch_size);
		if (synth_batch_size != null && synth_batch_size >= 1) out.synth_batch_size = synth_batch_size;
		if (r.task_type) out.task_type = String(r.task_type);
		if (r.track) out.track = String(r.track);
		if (r.synth_model) out.synth_model = String(r.synth_model);
		if (r.lm_model) out.lm_model = String(r.lm_model);
		if (r.lora && loraList.includes(String(r.lora))) out.lora = String(r.lora);
		const lora_scale = num(r.lora_scale);
		if (lora_scale != null) out.lora_scale = lora_scale;
		return out;
	}

	// save current form edits back into pendingRequests[pendingIndex]
	function savePending() {
		if (app.pendingRequests.length > 0 && app.pendingIndex < app.pendingRequests.length) {
			app.pendingRequests[app.pendingIndex] = buildRequest();
		}
	}

	// load pendingRequests[index] into the form.
	// synth params are form-global, not per-pending: preserve them across switches.
	function loadPending(index: number) {
		const r = app.pendingRequests[index];
		setRequest({
			...r,
			inference_steps: app.request.inference_steps,
			guidance_scale: app.request.guidance_scale,
			shift: app.request.shift,
			seed: app.request.seed,
			audio_cover_strength: app.request.audio_cover_strength,
			cover_noise_strength: app.request.cover_noise_strength,
			synth_batch_size: app.request.synth_batch_size
		});
		app.pendingIndex = index;
	}

	// switch to a different pending composition (saves current edits first)
	function switchPending(delta: number) {
		const next = app.pendingIndex + delta;
		if (next < 0 || next >= app.pendingRequests.length) return;
		savePending();
		loadPending(next);
	}

	// shared: call an LM endpoint and load results into the form.
	// preserves synth params (steps, CFG, shift, seed, batch, cover strength).
	async function lmCall(fn: (req: AceRequest) => Promise<AceRequest[]>) {
		busy = true;
		try {
			const req = buildRequest();
			req.audio_codes = '';
			const results = await fn(req);
			if (results.length > 0) {
				app.pendingRequests = results;
				app.pendingIndex = 0;
				setRequest({
					...results[0],
					inference_steps: app.request.inference_steps,
					guidance_scale: app.request.guidance_scale,
					shift: app.request.shift,
					seed: app.request.seed,
					audio_cover_strength: app.request.audio_cover_strength,
					cover_noise_strength: app.request.cover_noise_strength,
					synth_batch_size: app.request.synth_batch_size
				});
			}
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			busy = false;
		}
	}

	// Inspire: short caption -> fresh metadata + lyrics (no audio codes)
	async function inspire() {
		await lmCall(lmInspire);
	}

	// Format: caption + lyrics -> metadata + lyrics (no audio codes)
	async function format() {
		await lmCall(lmFormat);
	}

	// Compose: send form to LM, store all enriched results for batch synth.
	// The LM preserves user-provided fields and fills the rest independently
	// per batch item. Each result is a complete standalone request.
	async function compose() {
		await lmCall(lmGenerate);
	}

	// POST /synth: send pending requests (or current form) to the server.
	// synth params (batch, seed, steps, CFG, shift) come from the form, not from pending.
	// server groups by request and expands synth_batch_size for GPU batching.
	// webui resolves seeds and predicts the expanded list for SongCard mapping.
	async function synthesize() {
		busy = true;
		try {
			savePending();
			const reqs: AceRequest[] =
				app.pendingRequests.length > 0 ? $state.snapshot(app.pendingRequests) : [buildRequest()];

			// read synth params from the form (global, not per-pending).
			const synthBatch = Math.max(1, Number(app.request.synth_batch_size) || 1);
			const userSeed = Number(app.request.seed);
			const hasSeed = Number.isFinite(userSeed) && userSeed >= 0;
			const synthParams: Partial<AceRequest> = {};
			const steps = num(app.request.inference_steps);
			if (steps != null) synthParams.inference_steps = steps;
			const cfg = num(app.request.guidance_scale);
			if (cfg != null) synthParams.guidance_scale = cfg;
			const sh = num(app.request.shift);
			if (sh != null) synthParams.shift = sh;
			const acs = num(app.request.audio_cover_strength);
			if (acs != null) synthParams.audio_cover_strength = acs;
			const cns = num(app.request.cover_noise_strength);
			if (cns != null) synthParams.cover_noise_strength = cns;
			const rps = num(app.request.repaint_strength);
			if (rps != null) synthParams.repaint_strength = rps;
			// task_type and track from form
			const t = app.request.task_type || '';
			if (t) synthParams.task_type = t;
			if (app.request.track) synthParams.track = app.request.track;
			// model routing from form
			if (app.request.synth_model) synthParams.synth_model = app.request.synth_model;
			if (app.request.lora && loraList.includes(String(app.request.lora)))
				synthParams.lora = app.request.lora;
			const loraScale = num(app.request.lora_scale);
			if (loraScale != null) synthParams.lora_scale = loraScale;
			// repaint/lego: inject range from source audio selection (optional for lego)
			if (
				(t === TASK_REPAINT || t === TASK_LEGO) &&
				app.srcRangeStart >= 0 &&
				app.srcRangeEnd > app.srcRangeStart
			) {
				synthParams.repainting_start = app.srcRangeStart;
				synthParams.repainting_end = app.srcRangeEnd;
			}

			// resolve seeds, build server payload and local expanded list for SongCard mapping.
			// server receives synth_batch_size and expands internally (groups by T for GPU batch).
			// webui predicts the same expansion: seed, seed+1, ..., seed+N-1.
			const toSend: AceRequest[] = [];
			const expanded: AceRequest[] = [];
			for (const r of reqs) {
				const base = hasSeed ? userSeed : Math.floor(Math.random() * 0x100000000);
				toSend.push({ ...r, ...synthParams, seed: base, synth_batch_size: synthBatch });
				for (let i = 0; i < synthBatch; i++) {
					expanded.push({ ...r, ...synthParams, seed: base + i, synth_batch_size: 1 });
				}
			}

			// find source audio (cover/lego/repaint) and reference audio (timbre)
			const srcSong = app.srcSongId != null ? app.songs.find((s) => s.id === app.srcSongId) : null;
			const refSong = app.refSongId != null ? app.songs.find((s) => s.id === app.refSongId) : null;

			const blobs =
				srcSong || refSong
					? await synthGenerateWithAudio(
							toSend,
							srcSong?.audio ?? null,
							refSong?.audio ?? null,
							app.format
						)
					: await synthGenerate(toSend, app.format);
			const now = Date.now();
			const baseName = app.name || 'Untitled';
			for (let i = blobs.length - 1; i >= 0; i--) {
				const r = expanded[i];
				const song = {
					name: baseName,
					format: app.format,
					created: now + i,
					caption: r.caption,
					seed: r.seed || 0,
					duration: r.duration || 0,
					request: r,
					audio: blobs[i]
				} as Song;
				song.id = await putSong(song);
				app.songs.unshift(song);
			}
			app.pendingRequests = [];
			app.pendingIndex = 0;
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			busy = false;
		}
	}

	function ph(v: unknown): string {
		return v != null ? String(v) : '';
	}
</script>

<form class="request-form" onsubmit={(e) => e.preventDefault()}>
	<input
		type="file"
		accept=".json,.mp3,.wav"
		bind:this={fileInput}
		onchange={onFileSelected}
		hidden
	/>
	<div class="toolbar">
		<button type="button" onclick={importJson} title="Open JSON prompt, MP3 or WAV"
			><FolderOpen size={14} /> Open</button
		>
		<button type="button" onclick={exportJson} title="Save JSON prompt"
			><Download size={14} /> Save</button
		>
		<button type="button" onclick={reset} title="Reset prompt"><RotateCcw size={14} /> Reset</button
		>
	</div>

	<details>
		<summary>Models</summary>
		<div class="details-body">
			<div class="model-row">
				<span class="model-label">LM</span>
				<select bind:value={app.request.lm_model}>
					{#each lmModels as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">DiT</span>
				<select bind:value={app.request.synth_model}>
					{#each ditModels as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">LoRA</span>
				<select bind:value={app.request.lora}>
					<option value="">Disabled</option>
					{#if loraStale}
						<option value={app.request.lora} disabled>{app.request.lora}</option>
					{/if}
					{#each loraList as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
				<input
					type="text"
					class="batch-input"
					placeholder="1.0"
					bind:value={app.request.lora_scale}
				/>
			</div>
		</div>
	</details>

	<label class="section-label"
		>Name
		<input type="text" bind:value={app.name} placeholder="Untitled" />
	</label>

	<label class="section-label"
		>Caption
		<textarea
			rows="8"
			placeholder="Upbeat pop rock with driving guitars... (the only required field, may be enriched by the LM)"
			bind:value={app.request.caption}
		></textarea>
	</label>

	<label class="section-label"
		>Lyrics
		<textarea
			rows="8"
			placeholder="Write your own lyrics, type [Instrumental], or leave empty to let the LM create them..."
			bind:value={app.request.lyrics}
		></textarea>
	</label>

	<details open>
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
			</div>
		</div>
	</details>

	<div class="lm-row">
		<button type="button" disabled={busy} onclick={inspire}>Inspire</button>
		<button type="button" disabled={busy} onclick={format}>Format</button>
	</div>

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
			<label
				>Negative prompt
				<textarea
					rows="4"
					placeholder="Styles or instruments to steer away from, e.g. saxophone, autotune, screaming, low quality..."
					bind:value={app.request.lm_negative_prompt}
				></textarea>
			</label>
			<label
				>Audio codes
				<textarea
					rows="4"
					placeholder="Filled by Compose, or paste for dit-only"
					bind:value={app.request.audio_codes}
				></textarea>
			</label>
		</div>
	</details>

	<div class="selector-row">
		<label class="selector-label"
			>Batch <input
				type="number"
				class="batch-input"
				min="1"
				max={app.props?.cli?.max_batch || 9}
				bind:value={app.request.lm_batch_size}
			/></label
		>
		<span class="selector-label">Pending</span>
		<div class="pending-nav">
			<button type="button" class="nav-btn" onclick={() => switchPending(-1)}>&lt;</button>
			<span class="nav-label"
				>{app.pendingRequests.length > 0 ? app.pendingIndex + 1 : 0} / {app.pendingRequests
					.length}</span
			>
			<button type="button" class="nav-btn" onclick={() => switchPending(1)}>&gt;</button>
		</div>
	</div>

	<button type="button" disabled={busy} onclick={compose}>Compose</button>

	<details open>
		<summary>Flow matching parameters</summary>
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
					>Cover strength <input
						type="text"
						placeholder={ph(d?.audio_cover_strength)}
						bind:value={app.request.audio_cover_strength}
					/></label
				>
				<label
					>Cover noise <input
						type="text"
						placeholder={ph(d?.cover_noise_strength)}
						bind:value={app.request.cover_noise_strength}
					/></label
				>
				<label
					>Repaint strength <input
						type="text"
						placeholder={ph(d?.repaint_strength)}
						bind:value={app.request.repaint_strength}
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
				<label
					>Seed <input type="text" placeholder={ph(d?.seed)} bind:value={app.request.seed} /></label
				>
			</div>
		</div>
	</details>

	<div class="selector-row">
		<label class="selector-label"
			>Task <select
				class="batch-input"
				value={taskType}
				onchange={(e) => {
					app.request.task_type = e.currentTarget.value;
				}}
			>
				<option value="">text2music</option>
				<option value={TASK_COVER}>cover</option>
				<option value={TASK_REPAINT}>repaint</option>
				<option value={TASK_LEGO}>lego</option>
				<option value={TASK_EXTRACT}>extract</option>
				<option value={TASK_COMPLETE}>complete</option>
			</select></label
		>
		<label class="selector-label"
			>Track <select class="batch-input" bind:value={app.request.track}>
				{#each TRACK_NAMES as name}
					<option value={name}>{name}</option>
				{/each}
			</select></label
		>
	</div>

	<div class="selector-row">
		<label class="selector-label"
			>Batch <input
				type="number"
				class="batch-input"
				min="1"
				max="9"
				bind:value={app.request.synth_batch_size}
			/></label
		>
		<span class="selector-label">Format</span>
		<label class="selector-label">
			<input type="radio" name="format" value="mp3" bind:group={app.format} /> MP3
		</label>
		<label class="selector-label">
			<input type="radio" name="format" value="wav" bind:group={app.format} /> WAV
		</label>
	</div>

	<button type="button" disabled={busy} onclick={synthesize}>Synthesize</button>
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
	.section-label {
		color: var(--fg);
		font-weight: 600;
	}
	textarea,
	input[type='text'],
	select {
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
		color: var(--fg);
		font-weight: 600;
		padding: 0.4rem 0;
	}
	details summary:hover {
		color: var(--fg);
	}
	.details-body {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.25rem 0 0.5rem;
	}
	.model-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.model-label {
		font-size: 0.85rem;
		color: var(--fg-dim);
		flex-shrink: 0;
		width: 2rem;
	}
	.model-row select {
		flex: 1;
		min-width: 0;
	}
	.selector-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.selector-label {
		flex-direction: row;
		align-items: center;
		gap: 0.3rem;
		cursor: pointer;
		font-size: 0.85rem;
		color: var(--fg-dim);
	}
	.batch-input {
		padding: 0.2rem 0.3rem;
		font-size: 0.8rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-input);
		color: var(--fg);
	}
	input.batch-input {
		width: 3rem;
		text-align: center;
	}
	.pending-nav {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.nav-btn {
		padding: 0.15rem 0.4rem !important;
		font-size: 0.75rem !important;
		min-width: 0 !important;
	}
	.nav-label {
		font-size: 0.75rem;
		font-family: monospace;
		color: var(--fg);
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
	}
	.lm-row {
		display: flex;
		gap: 0.5rem;
	}
	.lm-row button {
		flex: 1;
	}
</style>

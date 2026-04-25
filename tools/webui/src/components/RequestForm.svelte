<script lang="ts">
	import { onMount } from 'svelte';
	import { RotateCcw, Download, FolderOpen, X } from '@lucide/svelte';
	import { app, toast, setRequest } from '../lib/state.svelte.js';
	import { rollDice } from '../lib/dice.js';
	import {
		lmSubmit,
		lmSubmitInspire,
		lmSubmitFormat,
		synthSubmit,
		synthSubmitWithAudio,
		pollJob,
		jobResultJson,
		jobResultBlobs,
		vaeDecode,
		cancelJob
	} from '../lib/api.js';
	import { putSong, getAllSongs, saveJob, loadJob, loadJobId, clearJob } from '../lib/db.js';
	import {
		TASK_TEXT2MUSIC,
		TASK_COVER,
		TASK_COVER_NOFSQ,
		TASK_REPAINT,
		TASK_LEGO,
		TASK_EXTRACT,
		TASK_COMPLETE,
		INFER_ODE,
		INFER_SDE,
		DCW_MODE_LOW,
		DCW_MODE_HIGH,
		DCW_MODE_DOUBLE,
		DCW_MODE_PIX,
		TRACK_NAMES
	} from '../lib/config.js';
	import {
		num,
		buildSparse,
		clearSection,
		withCurrentSettings,
		pickSections
	} from '../lib/fields.js';
	import type { AceRequest, Song } from '../lib/types.js';

	let busyLm = $state(false);
	let busySynth = $state(false);
	let busy = $derived(busyLm || busySynth);
	let fileInput: HTMLInputElement;

	let d = $derived(app.props?.default);
	let ditModels = $derived(app.props?.models.dit ?? []);
	let lmModels = $derived(app.props?.models.lm ?? []);
	let adapterList = $derived(app.props?.adapters ?? []);
	let adapterStale = $derived(
		!!app.request.adapter && !adapterList.includes(String(app.request.adapter))
	);
	let vaeList = $derived(app.props?.models.vae ?? []);
	let taskType = $derived(app.request.task_type || d?.task_type || '');
	let dp = $derived(
		app.props?.presets
			? String(app.request.synth_model || '').includes('turbo')
				? app.props.presets.turbo
				: app.props.presets.sft
			: null
	);
	let needsTrack = $derived(
		taskType === TASK_LEGO || taskType === TASK_EXTRACT || taskType === TASK_COMPLETE
	);
	let singleTrack = $derived(taskType === TASK_LEGO || taskType === TASK_EXTRACT);

	// DiT input indicators
	let hasCodes = $derived(!!app.request.audio_codes?.trim() && app.srcSongId == null);
	let hasSrc = $derived(app.srcSongId != null);
	let hasRange = $derived(app.srcRangeStart != null || app.srcRangeEnd != null);
	let hasRef = $derived(app.refSongId != null);

	// instrumental mode: checked when lyrics and language match the convention.
	// any manual edit to either field naturally unchecks via $derived.
	let instrumental = $derived(
		String(app.request.lyrics || '').trim() === '[Instrumental]' &&
			String(app.request.vocal_language || '').trim() === 'unknown'
	);

	function toggleInstrumental() {
		if (instrumental) {
			app.request.lyrics = '';
			app.request.vocal_language = '';
		} else {
			app.request.lyrics = '[Instrumental]';
			app.request.vocal_language = 'unknown';
		}
	}

	// track selection: radio for lego/extract, multi for complete
	let selectedTracks: Set<string> = $state(new Set());

	function toggleTrack(name: string) {
		let next = new Set(selectedTracks);
		if (next.has(name)) {
			next.delete(name);
		} else {
			if (singleTrack) next.clear();
			next.add(name);
		}
		selectedTracks = next;
	}

	// sync set to request string (preserve TRACK_NAMES order)
	$effect(() => {
		app.request.track = TRACK_NAMES.filter((n: string) => selectedTracks.has(n)).join(' | ');
	});

	// clear tracks when task has no use for them, trim to 1 for radio modes
	$effect(() => {
		if (!needsTrack) {
			if (selectedTracks.size > 0) selectedTracks = new Set();
		} else if (singleTrack && selectedTracks.size > 1) {
			selectedTracks = new Set([...selectedTracks].slice(0, 1));
		}
	});

	// cancel the active pipeline job
	async function cancelPipeline() {
		try {
			if (busySynth) {
				const synthId = loadJobId('synth');
				if (synthId) await cancelJob(synthId);
			} else if (busyLm) {
				const lmId = loadJobId('lm');
				if (lmId) await cancelJob(lmId);
			}
		} catch {}
	}

	// on mount: resume polling for any pending jobs in localStorage.
	onMount(() => {
		// LM job
		const lmId = loadJobId('lm');
		if (lmId) {
			busyLm = true;
			pollJob(lmId)
				.then(() => jobResultJson(lmId))
				.then((results) => {
					clearJob('lm');
					app.pendingRequests = results;
					app.pendingIndex = 0;
					if (results.length > 0) {
						setRequest(results[0]);
					}
				})
				.catch(() => {
					clearJob('lm');
				})
				.finally(() => {
					busyLm = false;
				});
		}

		// synth job
		const synthJob = loadJob('synth');
		if (synthJob) {
			busySynth = true;
			pollJob(synthJob.id)
				.then(() => jobResultBlobs(synthJob.id))
				.then(async ({ audios, latents }) => {
					clearJob('synth');
					const now = Date.now();
					for (let i = audios.length - 1; i >= 0; i--) {
						const t = synthJob.tracks[i] || {
							caption: '',
							seed: 0,
							duration: 0,
							task: '',
							request: { caption: '' }
						};
						const song: Song = {
							name: synthJob.name,
							format: synthJob.format,
							created: now + i,
							caption: t.caption,
							seed: t.seed,
							duration: t.duration,
							request: t.request,
							audio: audios[i],
							latents: latents[i]
						};
						await putSong(song);
					}
					app.songs = (await getAllSongs()).reverse();
				})
				.catch(() => {
					clearJob('synth');
				})
				.finally(() => {
					busySynth = false;
				});
		}
	});

	function reset() {
		app.name = '';
		setRequest({ caption: '' });
		app.pendingRequests = [];
		app.pendingIndex = 0;
		selectedTracks = new Set();
	}

	function exportJson() {
		const json = JSON.stringify(buildRequest(), null, 2);
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		const safe = app.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'request';
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

		// MP3 or WAV: create song card (audio only, use Scan for metadata)
		if (ext === 'mp3' || ext === 'wav') {
			openAudio(file, ext);
			return;
		}

		// VAE latents: decode through /vae to obtain the audio, create a
		// card already populated with both blobs (latents badge lit on day one).
		if (ext === 'vae') {
			openLatents(file);
			return;
		}

		toast('Unsupported file type: ' + ext);
	}

	// open audio file: create song card with audio only (no server call).
	// use Scan on the card to analyze metadata.
	async function openAudio(file: File, ext: string) {
		const blob = new Blob([await file.arrayBuffer()], {
			type: ext === 'wav' ? 'audio/wav' : 'audio/mpeg'
		});
		const name = file.name.replace(/\.(mp3|wav)$/i, '') || 'Imported';
		const song: Song = {
			name,
			format: ext,
			created: Date.now(),
			caption: '',
			seed: 0,
			duration: 0,
			request: { caption: '' },
			audio: blob
		};
		song.id = await putSong(song);
		app.songs.unshift(song);
		app.name = name;
		toast('Opened: ' + name, 4000, true);
	}

	// open VAE latents file: validate framing client-side, post to /vae
	// to render the audio, then create a card with both blobs ready to use.
	// Same final shape as openAudio plus the latents attached, so cover-nofsq
	// against this card skips the VAE encode from the very first run.
	async function openLatents(file: File) {
		const buf = await file.arrayBuffer();
		if (buf.byteLength === 0 || buf.byteLength % 256 !== 0) {
			toast('Invalid .vae file: size must be a multiple of 256 bytes (64 channels x f32)');
			return;
		}
		const T = buf.byteLength / 256;
		if (T > 15000) {
			toast('Invalid .vae file: too long (max 15000 frames = 10 min)');
			return;
		}
		const latentsBlob = new Blob([buf], { type: 'application/octet-stream' });
		const name = file.name.replace(/\.vae$/i, '') || 'Imported';
		try {
			const jobId = await vaeDecode(latentsBlob, app.request);
			await pollJob(jobId);
			const { audios } = await jobResultBlobs(jobId);
			if (!audios.length) throw new Error('Decode returned no audio');
			const song: Song = {
				name,
				format: app.format,
				created: Date.now(),
				caption: '',
				seed: 0,
				duration: 0,
				request: { caption: '' },
				audio: audios[0],
				latents: latentsBlob
			};
			song.id = await putSong(song);
			app.songs.unshift(song);
			app.name = name;
			toast('Opened: ' + name, 4000, true);
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		}
	}

	// snapshot app.request into a clean AceRequest with proper types.
	// bind:value guarantees app.request always matches the DOM.
	function buildRequest(): AceRequest {
		const out = buildSparse(app.request);
		if (out.adapter && !adapterList.includes(String(out.adapter))) delete out.adapter;
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
		setRequest(withCurrentSettings(r, app.request));
		app.pendingIndex = index;
	}

	// switch to a different pending composition (saves current edits first)
	function switchPending(delta: number) {
		const next = app.pendingIndex + delta;
		if (next < 0 || next >= app.pendingRequests.length) return;
		savePending();
		loadPending(next);
	}

	// shared: call an LM endpoint via the job system and load results into the form.
	async function lmCall(fn: (req: AceRequest) => Promise<string>) {
		busyLm = true;
		try {
			const req = buildRequest();
			req.audio_codes = '';
			const jobId = await fn(req);
			saveJob('lm', jobId);
			await pollJob(jobId);
			const results = await jobResultJson(jobId);
			clearJob('lm');
			if (results.length > 0) {
				app.pendingRequests = results;
				app.pendingIndex = 0;
				setRequest(withCurrentSettings(results[0], app.request));
			}
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			busyLm = false;
		}
	}

	// Dice: pick a random example prompt and fill the caption
	function dice() {
		setRequest(rollDice());
	}

	// Inspire: short caption -> fresh metadata + lyrics (no audio codes)
	async function inspire() {
		await lmCall(lmSubmitInspire);
	}

	// Format: caption + lyrics -> metadata + lyrics (no audio codes)
	async function format() {
		await lmCall(lmSubmitFormat);
	}

	// Compose: send form to LM, store all enriched results for batch synth.
	// The LM preserves user-provided fields and fills the rest independently
	// per batch item. Each result is a complete standalone request.
	async function compose() {
		await lmCall(lmSubmit);
	}

	// POST /synth: send pending requests (or current form) to the server.
	// synth params (batch, seed, steps, CFG, shift) come from the form, not from pending.
	// server groups by request and expands synth_batch_size for GPU batching.
	// webui resolves seeds and predicts the expanded list for SongCard mapping.
	async function synthesize() {
		busySynth = true;
		try {
			savePending();
			const reqs: AceRequest[] =
				app.pendingRequests.length > 0 ? $state.snapshot(app.pendingRequests) : [buildRequest()];

			// read synth params from the form (global, not per-pending).
			const synthBatch = Math.max(1, Number(app.request.synth_batch_size) || 1);
			const userSeed = num(app.request.seed);
			const hasSeed = userSeed != null && userSeed >= 0;
			const synthParams = pickSections(app.request, ['flow', 'advanced', 'toolbar', 'routing']);
			// seed and synth_batch_size are per-expansion, handled below
			delete synthParams.seed;
			delete synthParams.synth_batch_size;
			if (synthParams.adapter && !adapterList.includes(String(synthParams.adapter)))
				delete synthParams.adapter;

			// resolve seeds, build server payload and local expanded list for SongCard mapping.
			// server receives synth_batch_size and expands internally (groups by T for GPU batch).
			// webui predicts the same expansion: seed, seed+1, ..., seed+N-1.
			const toSend: AceRequest[] = [];
			const expanded: AceRequest[] = [];
			for (const r of reqs) {
				const base = hasSeed ? userSeed : Math.floor(Math.random() * 0x100000000);
				toSend.push({ ...r, ...synthParams, seed: base, synth_batch_size: synthBatch });
				for (let i = 0; i < synthBatch; i++) {
					expanded.push({ ...r, ...synthParams, seed: base + i });
				}
			}

			// find source audio (cover/lego/repaint) and reference audio (timbre)
			const srcSong = app.srcSongId != null ? app.songs.find((s) => s.id === app.srcSongId) : null;
			const refSong = app.refSongId != null ? app.songs.find((s) => s.id === app.refSongId) : null;

			// extract DiT variant from model filename
			// "acestep-v15-xl-turbo-Q8_0.gguf" -> "xl-turbo"
			const model = String(app.request.synth_model || '');
			const vm = model.match(/^acestep-v15-(.+?)-(Q\d.*|BF16)\.gguf$/);
			const variant = vm ? vm[1] : '';
			const baseName = app.name || 'Untitled';

			// submit job, poll until done, fetch result. When the source song
			// or timbre reference already carries cached latents, we upload
			// those instead of the audio: the server skips the matching VAE
			// encode entirely.
			const jobId =
				srcSong || refSong
					? await synthSubmitWithAudio(
							toSend,
							srcSong?.latents ? null : (srcSong?.audio ?? null),
							srcSong?.latents ?? null,
							refSong?.latents ? null : (refSong?.audio ?? null),
							refSong?.latents ?? null,
							app.format
						)
					: await synthSubmit(toSend, app.format);
			saveJob('synth', {
				id: jobId,
				name: baseName,
				format: app.format,
				variant,
				tracks: expanded.map((r) => ({
					caption: r.caption || '',
					seed: r.seed || 0,
					duration: r.duration || 0,
					task: r.task_type || 'text2music',
					request: r
				}))
			});
			await pollJob(jobId);
			const { audios, latents } = await jobResultBlobs(jobId);
			clearJob('synth');

			const now = Date.now();

			for (let i = audios.length - 1; i >= 0; i--) {
				const r = expanded[i];
				const song = {
					name: baseName,
					format: app.format,
					created: now + i,
					caption: r.caption,
					seed: r.seed || 0,
					duration: r.duration || 0,
					request: r,
					audio: audios[i],
					latents: latents[i]
				} as Song;
				song.id = await putSong(song);
				app.songs.unshift(song);
			}
			app.pendingRequests = [];
			app.pendingIndex = 0;
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			busySynth = false;
		}
	}

	function clearMetadata() {
		clearSection(app.request, 'metadata');
	}

	function clearFlowMatching() {
		clearSection(app.request, 'flow');
		app.srcRangeStart = null;
		app.srcRangeEnd = null;
	}

	function clearAdvanced() {
		clearSection(app.request, 'advanced');
	}

	function clearAdvancedLm() {
		clearSection(app.request, 'lm');
	}

	function ph(v: unknown): string {
		return v != null ? String(v) : '';
	}
</script>

<form class="request-form" onsubmit={(e) => e.preventDefault()}>
	<input
		type="file"
		accept=".json,.mp3,.wav,.vae"
		bind:this={fileInput}
		onchange={onFileSelected}
		hidden
	/>
	<div class="toolbar">
		<button type="button" onclick={importJson} title="Open JSON prompt, MP3, WAV or VAE latents"
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
				<select
					class="model-select"
					bind:value={app.request.lm_model}
					title="Language Model for Inspire, Format and Compose. Scanned from --models directory at startup."
				>
					{#each lmModels as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">DiT</span>
				<select
					class="model-select"
					bind:value={app.request.synth_model}
					title="Diffusion Transformer for Synthesize. Scanned from --models directory at startup."
				>
					{#each ditModels as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
			<div class="model-row">
				<span class="model-label">LoRA</span>
				<select
					class="model-select"
					bind:value={app.request.adapter}
					title="Adapter merged into DiT at load time. Must match the exact DiT it was trained on. Scanned from --adapters directory. Supports LoRA as a ComfyUI single .safetensors or a PEFT directory with adapter_model.safetensors and adapter_config.json."
				>
					<option value="">Disabled</option>
					{#if adapterStale}
						<option value={app.request.adapter} disabled>{app.request.adapter}</option>
					{/if}
					{#each adapterList as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
				<input
					type="text"
					class="batch-input"
					placeholder="1.0"
					bind:value={app.request.adapter_scale}
					title="Adapter scale factor. Lower if you hear structured noise or artifacts. Raise for stronger effect."
				/>
			</div>
			<div class="model-row">
				<span class="model-label">VAE</span>
				<select
					class="model-select"
					bind:value={app.request.vae}
					title="Variational Auto-Encoder for audio <-> latent conversion. Scanned from --models directory at startup."
				>
					{#each vaeList as name}
						<option value={name}>{name}</option>
					{/each}
				</select>
			</div>
		</div>
	</details>

	<div class="section-title">Name</div>
	<input type="text" bind:value={app.name} placeholder="Untitled" />

	<div class="section-title caption-header">
		Caption
		<label
			class="header-toggle"
			title="Lock caption: LM keeps your text intact (skips CoT caption refinement)"
		>
			<input
				type="checkbox"
				checked={!app.request.use_cot_caption}
				onchange={(e) => (app.request.use_cot_caption = !e.currentTarget.checked)}
			/> Lock
		</label>
	</div>
	<textarea
		rows="8"
		placeholder="Upbeat pop rock with driving guitars... (the only required field, enriched by the LM unless all prompt fields are filled.)"
		bind:value={app.request.caption}
	></textarea>

	<div class="section-title lyrics-header">
		Lyrics
		<label class="header-toggle" title="Set lyrics to [Instrumental] and language to unknown">
			<input type="checkbox" checked={instrumental} onchange={toggleInstrumental} /> Instrumental
		</label>
	</div>
	<textarea
		rows="8"
		placeholder="Write your own lyrics or leave empty to let the LM create them..."
		bind:value={app.request.lyrics}
	></textarea>

	<div class="section-title metadata-header">
		Metadata
		<button
			type="button"
			class="clear-btn"
			title="Clear metadata"
			onclick={clearMetadata}
			aria-label="Clear metadata"
		>
			<X size={20} />
		</button>
	</div>
	<div class="meta-grid">
		<label
			>Language <input
				type="text"
				placeholder={ph(d?.vocal_language)}
				bind:value={app.request.vocal_language}
			/></label
		>
		<label>BPM <input type="text" placeholder={ph(d?.bpm)} bind:value={app.request.bpm} /></label>
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
			>LM seed <input
				type="text"
				placeholder={ph(d?.lm_seed)}
				bind:value={app.request.lm_seed}
			/></label
		>
	</div>

	<div class="lm-row">
		<button
			type="button"
			disabled={busy}
			onclick={dice}
			title="Pick a random example from ACE-Step sample prompts. Use Inspire next to complete missing fields."
			>Dice</button
		>
		<button
			type="button"
			disabled={busy}
			onclick={inspire}
			title="Step 1: LM inference to generate metadata and lyrics from your caption. Next: Compose."
			>Inspire</button
		>
		<button
			type="button"
			disabled={busy}
			onclick={format}
			title="Step 1: LM inference to format existing lyrics for better generation quality. Next: Compose."
			>Format</button
		>
	</div>

	<details class="has-clear">
		<summary>Advanced LM</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear advanced LM"
			onclick={clearAdvancedLm}
			aria-label="Clear advanced LM"
		>
			<X size={20} />
		</button>
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
					placeholder="Filled by Compose. Do not edit unless you know what you are doing."
					bind:value={app.request.audio_codes}
				></textarea>
			</label>
		</div>
	</details>

	<div class="model-row">
		<span class="model-label">Batch</span>
		<input
			type="number"
			class="batch-input"
			min="1"
			max={app.props?.cli?.max_batch || 9}
			placeholder={ph(d?.lm_batch_size)}
			bind:value={app.request.lm_batch_size}
			title="Number of LM variations per Compose. Server must be started with --max-batch N (default 1). Higher values use more VRAM for the KV cache."
		/>
		<span class="spacer"></span>
		<span class="row-label">Pending</span>
		<div class="pending-nav">
			<button
				type="button"
				class="nav-btn"
				onclick={() => switchPending(-1)}
				title="Previous pending variation">&lt;</button
			>
			<span class="nav-label"
				>{app.pendingRequests.length > 0 ? app.pendingIndex + 1 : 0} / {app.pendingRequests
					.length}</span
			>
			<button
				type="button"
				class="nav-btn"
				onclick={() => switchPending(1)}
				title="Next pending variation">&gt;</button
			>
		</div>
	</div>

	<div class="action-row">
		<button
			type="button"
			disabled={busy}
			onclick={compose}
			title="Step 2: LM inference to generate audio codes that drive the flow matching. Next: Synthesize."
			>Compose</button
		>
		<button
			type="button"
			disabled={!busyLm}
			onclick={cancelPipeline}
			title="Cancel the active LM job">Cancel</button
		>
	</div>

	<details open>
		<summary>Task</summary>
		<div class="details-body">
			<div class="model-row">
				<span class="model-label">Type</span>
				<select
					class="model-select"
					value={taskType}
					onchange={(e) => {
						app.request.task_type = e.currentTarget.value;
					}}
				>
					<option value={TASK_TEXT2MUSIC}>Text2Music: from prompt and LM codes</option>
					<option value={TASK_COVER}>Cover: reinterpret in a new style</option>
					<option value={TASK_COVER_NOFSQ}>Cover (no FSQ): closer to the original</option>
					<option value={TASK_REPAINT}>Repaint: regenerate a region</option>
					<option value={TASK_LEGO}>Lego: add a stem over backing audio</option>
					<option value={TASK_EXTRACT}>Extract: isolate one stem from a mix</option>
					<option value={TASK_COMPLETE}>Complete: auto-arrange around a partial track</option>
				</select>
			</div>
			<div class="model-row track-row">
				<span class="model-label">Track</span>
				<div class="track-grid">
					{#each TRACK_NAMES as name}
						<button
							type="button"
							class="track-pill"
							class:active={selectedTracks.has(name)}
							disabled={!needsTrack}
							onclick={() => toggleTrack(name)}>{name}</button
						>
					{/each}
				</div>
			</div>
		</div>
	</details>

	<details open class="has-clear">
		<summary>Flow matching parameters</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear flow matching parameters"
			onclick={clearFlowMatching}
			aria-label="Clear flow matching parameters"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<div class="meta-grid">
				<label
					>Steps <input
						type="text"
						placeholder={ph(dp?.inference_steps)}
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
					>Repaint start <input
						type="text"
						placeholder={ph(d?.repainting_start)}
						value={app.srcRangeStart != null ? Math.round(app.srcRangeStart * 100) / 100 : ''}
						oninput={(e) => {
							const s = e.currentTarget.value.trim();
							app.srcRangeStart =
								s === '' ? null : isNaN(Number(s)) ? app.srcRangeStart : Number(s);
						}}
					/></label
				>
				<label
					>Repaint end <input
						type="text"
						placeholder={ph(d?.repainting_end)}
						value={app.srcRangeEnd != null ? Math.round(app.srcRangeEnd * 100) / 100 : ''}
						oninput={(e) => {
							const s = e.currentTarget.value.trim();
							app.srcRangeEnd = s === '' ? null : isNaN(Number(s)) ? app.srcRangeEnd : Number(s);
						}}
					/></label
				>
				<label
					>CFG scale <input
						type="text"
						placeholder={ph(dp?.guidance_scale)}
						bind:value={app.request.guidance_scale}
					/></label
				>
				<label
					>Shift <input
						type="text"
						placeholder={ph(dp?.shift)}
						bind:value={app.request.shift}
					/></label
				>
				<label
					>Seed <input type="text" placeholder={ph(d?.seed)} bind:value={app.request.seed} /></label
				>
			</div>
		</div>
	</details>

	<details class="has-clear">
		<summary>Advanced and post-processing</summary>
		<button
			type="button"
			class="clear-btn details-clear"
			title="Clear advanced and post-processing"
			onclick={clearAdvanced}
			aria-label="Clear advanced and post-processing"
		>
			<X size={20} />
		</button>
		<div class="details-body">
			<label
				>Custom scheduler <input
					type="text"
					placeholder="Descending floats 1 -> 0, comma-separated"
					bind:value={app.request.custom_timesteps}
				/></label
			>
			<div class="meta-grid">
				<label
					>DCW mode <select
						value={app.request.dcw_mode || d?.dcw_mode || ''}
						onchange={(e) => {
							app.request.dcw_mode = e.currentTarget.value;
						}}
					>
						<option value={DCW_MODE_LOW}>Low</option>
						<option value={DCW_MODE_HIGH}>High</option>
						<option value={DCW_MODE_DOUBLE}>Double</option>
						<option value={DCW_MODE_PIX}>Pix</option>
					</select></label
				>
				<label
					>DCW scaler <input
						type="text"
						placeholder={ph(d?.dcw_scaler)}
						bind:value={app.request.dcw_scaler}
					/></label
				>
				<label
					>DCW high scaler <input
						type="text"
						placeholder={ph(d?.dcw_high_scaler)}
						bind:value={app.request.dcw_high_scaler}
					/></label
				>
				<label
					>Method <select
						value={app.request.infer_method || d?.infer_method || ''}
						onchange={(e) => {
							app.request.infer_method = e.currentTarget.value;
						}}
					>
						<option value={INFER_ODE}>ODE Euler</option>
						<option value={INFER_SDE}>SDE Stochastic</option>
					</select></label
				>
				<label
					>Latent shift <input
						type="text"
						placeholder={ph(d?.latent_shift)}
						bind:value={app.request.latent_shift}
					/></label
				>
				<label
					>Latent rescale <input
						type="text"
						placeholder={ph(d?.latent_rescale)}
						bind:value={app.request.latent_rescale}
					/></label
				>
				<label
					>Peak clip <input
						type="text"
						placeholder={ph(d?.peak_clip)}
						bind:value={app.request.peak_clip}
					/></label
				>
				<label
					>MP3 bitrate <input
						type="text"
						placeholder={ph(d?.mp3_bitrate)}
						bind:value={app.request.mp3_bitrate}
					/></label
				>
			</div>
		</div>
	</details>

	<div class="model-row">
		<span class="model-label">Batch</span>
		<input
			type="number"
			class="batch-input"
			min="1"
			max="9"
			placeholder={ph(d?.synth_batch_size)}
			bind:value={app.request.synth_batch_size}
			title="Number of DiT variations per request. Each uses a consecutive seed."
		/>
		<span class="spacer"></span>
		<span class="model-label">Format</span>
		<select
			bind:value={app.format}
			title="Output audio format. WAV32 outputs raw IEEE float without normalization."
		>
			<option value="mp3">MP3</option>
			<option value="wav16">WAV16</option>
			<option value="wav24">WAV24</option>
			<option value="wav32">WAV32</option>
		</select>
	</div>

	<div class="model-row cond-row">
		<span class="model-label">Cond</span>
		<div class="track-grid">
			<span class="dit-ind" class:on={hasCodes}>LM codes</span>
			<span class="dit-ind" class:on={hasSrc}>Src audio</span>
			<span class="dit-ind" class:on={hasRange}>Range</span>
			<span class="dit-ind" class:on={hasRef}>Timbre ref</span>
		</div>
	</div>

	<div class="action-row">
		<button
			type="button"
			disabled={busy}
			onclick={synthesize}
			title="Step 3: DiT flow matching + VAE decoding to synthesize audio from codes. No green indicators above (LM codes, Src audio, Timbre ref)? The DiT will hallucinate freely from your prompt: very creative, but unpredictable."
			>Synthesize</button
		>
		<button
			type="button"
			disabled={!busySynth}
			onclick={cancelPipeline}
			title="Cancel the active synth job">Cancel</button
		>
	</div>
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
	.section-title {
		font-size: 0.85rem;
		color: var(--fg);
		font-weight: 600;
		padding: 0.4rem 0 0;
	}
	.lyrics-header,
	.caption-header,
	.metadata-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.header-toggle {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.8rem;
		font-weight: 400;
		color: var(--fg-dim);
		cursor: pointer;
	}
	.header-toggle input[type='checkbox'] {
		cursor: pointer;
	}
	.has-clear {
		position: relative;
	}
	.details-clear {
		position: absolute;
		top: 0.4rem;
		right: 0;
	}
	.clear-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		border: none;
		background: transparent;
		color: var(--fg-dim);
		cursor: pointer;
		line-height: 0;
	}
	.clear-btn:hover {
		color: var(--fg);
	}
	textarea,
	input[type='text'],
	input[type='number'],
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
		min-width: 2rem;
	}
	.model-select {
		flex: 1;
		min-width: 0;
	}
	.spacer {
		flex: 1;
	}
	.row-label {
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
		padding: 0.4rem 0.4rem !important;
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
	.dit-ind {
		padding: 0.15rem 0.4rem;
		border-radius: 4px;
		font-size: 0.8rem;
		white-space: nowrap;
		background: var(--bg-err, #c0392b);
		color: #fff;
		opacity: 0.6;
		text-align: center;
		flex: 1;
	}
	.dit-ind.on {
		background: var(--bg-ok, #27ae60);
		opacity: 1;
	}
	.track-row,
	.cond-row {
		align-items: flex-start;
	}
	.track-row .model-label,
	.cond-row .model-label {
		padding-top: 0.2rem;
	}
	.track-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		flex: 1;
	}
	.track-pill {
		padding: 0.2rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		font-size: 0.8rem;
		font-family: inherit;
		cursor: pointer;
		background: var(--bg-input);
		color: var(--fg-dim);
		text-align: center;
		flex: 1;
		max-width: 33%;
	}
	.track-pill.active {
		background: var(--bg-btn-hover);
		color: var(--fg);
		border-color: var(--focus);
	}
	.lm-row {
		display: flex;
		gap: 0.5rem;
	}
	.lm-row button {
		flex: 1;
	}
	.action-row {
		display: flex;
		gap: 0.5rem;
	}
	.action-row button {
		flex: 1;
	}
</style>

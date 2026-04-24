<script lang="ts">
	import { Play, Square, Pencil, Ear, Download, Cpu, Trash2, ChevronDown } from '@lucide/svelte';
	import { app, setRequest, toast } from '../lib/state.svelte.js';
	import { deleteSong } from '../lib/db.js';
	import {
		understandSubmit,
		vaeEncode,
		pollJob,
		jobResultUnderstand,
		jobResultLatents
	} from '../lib/api.js';
	import { saveJob, clearJob, putSong } from '../lib/db.js';
	import type { Song } from '../lib/types.js';
	import Waveform from './Waveform.svelte';
	import Menu, { type MenuItem } from './Menu.svelte';

	let { song }: { song: Song } = $props();

	let playing = $state(false);
	let time = $state(0);
	let dur = $state(0);
	let rangeStart = $state(0);
	let rangeEnd = $state(0);

	let isRef = $derived(app.refSongId === song.id);
	let isSrc = $derived(app.srcSongId === song.id);

	function toggleRef() {
		if (isRef) {
			app.refSongId = null;
		} else {
			app.refSongId = song.id ?? null;
		}
	}

	function toggleSrc() {
		if (isSrc) {
			app.srcSongId = null;
			app.srcRangeStart = null;
			app.srcRangeEnd = null;
			rangeStart = 0;
			rangeEnd = 0;
		} else {
			app.srcSongId = song.id ?? null;
		}
	}

	// waveform drag to global state
	$effect(() => {
		if (isSrc && rangeEnd > rangeStart) {
			app.srcRangeStart = rangeStart;
			app.srcRangeEnd = rangeEnd;
		}
	});

	// global state to waveform visual (field input)
	$effect(() => {
		if (isSrc) {
			const rs = app.srcRangeStart;
			const re = app.srcRangeEnd;
			if (rs != null && re != null && re > rs) {
				rangeStart = rs;
				rangeEnd = re;
			} else {
				rangeStart = 0;
				rangeEnd = 0;
			}
		}
	});

	function toggle() {
		playing = !playing;
	}

	function load() {
		app.name = song.name;
		setRequest({ ...song.request });
		app.pendingRequests = [];
		app.pendingIndex = 0;
	}

	let scanning = $state(false);

	// analyze audio: send to /understand, fill form with detected metadata.
	// persists the job under 'lm' key so page reload resumes polling.
	// Uploads the cached cover latents when present so the server skips the
	// VAE encode. Understand is the canonical "complete this raw audio" op:
	// the source card is enriched in place with both outputs the server
	// produced for it, the latents that fed the FSQ tokenizer and the
	// detected metadata (caption, lyrics, codes, bpm, ...). Cover and synth
	// keep their "always create a new card" contract; only understand
	// touches an existing card and only the very one it analyzed.
	async function scan() {
		scanning = true;
		try {
			const jobId = await understandSubmit(
				song.latents ? null : song.audio,
				song.latents ?? null,
				app.request.lm_model as string,
				app.request.synth_model as string
			);
			saveJob('lm', jobId);
			await pollJob(jobId);
			const { requests, latents } = await jobResultUnderstand(jobId);
			clearJob('lm');
			if (song.id != null) {
				const newLatents = latents ?? song.latents ?? undefined;
				const newRequest =
					requests.length > 0 && !song.request.caption ? requests[0] : { ...song.request };
				const dirty = newLatents !== song.latents || newRequest !== song.request;
				if (dirty) {
					// Rebuild a plain object: $props in Svelte 5 are Proxies and
					// IndexedDB structuredClone refuses Proxies. Listing fields by
					// hand also avoids cloning the audio Blob, which is large.
					const enriched: Song = {
						id: song.id,
						name: song.name,
						format: song.format,
						created: song.created,
						caption: newRequest.caption ?? song.caption,
						seed: newRequest.seed ?? song.seed,
						duration: newRequest.duration ?? song.duration,
						request: newRequest,
						audio: song.audio,
						...(newLatents ? { latents: newLatents } : {})
					};
					await putSong(enriched);
					if (latents) song.latents = latents;
					if (newRequest !== song.request) song.request = newRequest;
				}
			}
			app.name = song.name;
			if (requests.length > 0) {
				setRequest(requests[0]);
			}
			app.pendingRequests = requests;
			app.pendingIndex = 0;
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			scanning = false;
		}
	}

	function downloadAudio() {
		const url = URL.createObjectURL(song.audio);
		const a = document.createElement('a');
		a.href = url;
		const safe = song.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'song';
		const ext = song.format.startsWith('wav') ? '.wav' : '.mp3';
		a.download = `${safe}${ext}`;
		a.click();
		URL.revokeObjectURL(url);
	}

	// Download the cached latents blob as a .vae file. Symmetric to
	// downloadAudio: the .vae plays back via Open (POST /vae decode path)
	// or feeds a future synth/understand call as src_latents, skipping the
	// VAE encode on reuse.
	function downloadLatents() {
		if (!song.latents) return;
		const url = URL.createObjectURL(song.latents);
		const a = document.createElement('a');
		a.href = url;
		const safe = song.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'song';
		a.download = `${safe}.vae`;
		a.click();
		URL.revokeObjectURL(url);
	}

	// VAE-only scan: POST /vae with the source audio (encode path), attach
	// the fresh latents to the card non-destructively. Plain-object rebuild
	// like scan() to dodge Svelte 5 Proxy + IndexedDB structuredClone, minus
	// the LM roundtrip. Ideal for priming a cover/repaint target without
	// paying the LM cost just to get the [VAE] badge lit.
	async function encodeOnly() {
		if (song.latents || song.id == null) return;
		scanning = true;
		try {
			const jobId = await vaeEncode(song.audio);
			saveJob('lm', jobId);
			await pollJob(jobId);
			const latents = await jobResultLatents(jobId);
			clearJob('lm');
			const enriched: Song = {
				id: song.id,
				name: song.name,
				format: song.format,
				created: song.created,
				caption: song.caption,
				seed: song.seed,
				duration: song.duration,
				request: { ...song.request },
				audio: song.audio,
				latents
			};
			await putSong(enriched);
			song.latents = latents;
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			scanning = false;
		}
	}

	async function remove() {
		if (song.id == null) return;
		if (app.refSongId === song.id) app.refSongId = null;
		if (app.srcSongId === song.id) app.srcSongId = null;
		await deleteSong(song.id);
		const idx = app.songs.findIndex((s) => s.id === song.id);
		if (idx >= 0) app.songs.splice(idx, 1);
	}

	// MM:SS:XX (hundredths) for current position
	function fmtPos(s: number): string {
		const m = Math.floor(s / 60);
		const sec = Math.floor(s % 60);
		const cs = Math.floor((s * 100) % 100);
		return (
			String(m).padStart(2, '0') +
			':' +
			String(sec).padStart(2, '0') +
			':' +
			String(cs).padStart(2, '0')
		);
	}

	// MM:SS for total duration
	function fmtDur(s: number): string {
		const m = Math.floor(s / 60);
		const sec = Math.floor(s % 60);
		return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
	}

	// Single action menu: one entry per user intent. Order mirrors a natural
	// flow (tweak prompt -> grab audio -> work with latents -> inspect -> destroy).
	// Delete relies on "open menu + pick Confirm" as a lightweight
	// confirmation step, no modal needed.
	const actionItems: MenuItem[] = $derived([
		{ icon: Pencil, label: 'Edit prompt', onSelect: load },
		{ icon: Download, label: 'Download audio', onSelect: downloadAudio },
		{ icon: Cpu, label: 'Compute VAE latents', onSelect: encodeOnly, disabled: !!song.latents },
		{
			icon: Download,
			label: 'Download VAE latents',
			onSelect: downloadLatents,
			disabled: !song.latents
		},
		{ icon: Ear, label: 'LM understand', onSelect: scan },
		{ icon: Trash2, label: 'Delete track', onSelect: remove }
	]);
</script>

<div class="card">
	<div class="card-header">
		<button class="icon-btn" onclick={toggle} title={playing ? 'Stop' : 'Play'}>
			{#if playing}
				<Square size={14} />
			{:else}
				<Play size={14} />
			{/if}
		</button>
		<span class="card-name">{song.name}</span>
		<Menu items={actionItems}>
			{#snippet trigger()}<ChevronDown size={14} /> Menu{/snippet}
		</Menu>
	</div>
	<Waveform
		audio={song.audio}
		bind:playing
		bind:time
		bind:dur
		selectable={isSrc}
		bind:rangeStart
		bind:rangeEnd
	/>
	<div class="card-footer">
		<span class="format-badge">{song.format.toUpperCase()}</span>
		{#if song.latents}
			<span class="format-badge">VAE</span>
		{/if}
		<span class="timecode">{fmtPos(time)} / {fmtDur(dur)}</span>
		<div class="card-actions">
			<label class="icon-btn"
				><input
					type="checkbox"
					class="ref-check"
					checked={isSrc}
					onchange={toggleSrc}
					title="Source audio"
				/> Src audio</label
			>
			<label class="icon-btn"
				><input
					type="checkbox"
					class="ref-check"
					checked={isRef}
					onchange={toggleRef}
					title="Timbre reference"
				/> Timbre ref</label
			>
		</div>
	</div>
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0.5rem;
		border: none;
		border-radius: 4px;
		background: var(--bg-card);
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.card-footer {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.card-name {
		font-size: 0.8rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		flex: 1;
	}
	.format-badge {
		font-size: 0.6rem;
		font-family: monospace;
		padding: 0.05rem 0.3rem;
		border-radius: 2px;
		background: var(--fg);
		color: var(--bg);
		flex-shrink: 0;
	}
	.timecode {
		font-size: 0.7rem;
		font-family: monospace;
		color: var(--fg);
		white-space: nowrap;
		flex: 1;
	}
	.card-actions {
		display: flex;
		align-items: center;
		gap: 0.2rem;
		flex-shrink: 0;
		font-size: 0.8rem;
	}
	.icon-btn {
		background: none;
		border: none;
		cursor: pointer;
		padding: 0.15rem;
		color: var(--fg);
		display: flex;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.8rem;
	}
	.icon-btn:hover {
		color: var(--focus);
	}
	.ref-check {
		cursor: pointer;
		accent-color: var(--focus);
	}
</style>

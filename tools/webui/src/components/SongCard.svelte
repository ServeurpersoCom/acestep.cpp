<script lang="ts">
	import {
		Play,
		Square,
		Pencil,
		Ear,
		Download,
		Cpu,
		Trash2,
		ChevronDown,
		Heart,
		Type,
		TriangleAlert
	} from '@lucide/svelte';
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
	import { displaySongName } from '../lib/songName.js';
	import Waveform from './Waveform.svelte';
	import Menu, { type MenuItem } from './Menu.svelte';
	import Dialog from './Dialog.svelte';

	let { song }: { song: Song } = $props();

	let playing = $state(false);
	let time = $state(0);
	let dur = $state(0);
	let rangeStart = $state(0);
	let rangeEnd = $state(0);

	let isRef = $derived(app.refSongId === song.id);
	let isSrc = $derived(app.srcSongId === song.id);

	// "(variant task)" suffix rebuilt from the request, used for the card
	// title and download filenames. Song.name itself stays the base name.
	let displayName = $derived(displaySongName(song));

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
	// Uploads the cached latents when present so the server skips the VAE
	// encode. Understand is the canonical "complete this raw audio" op:
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
				const useNewRequest = requests.length > 0 && !song.request.caption;
				const newRequest = useNewRequest ? requests[0] : song.request;
				const dirty = (latents && latents !== song.latents) || newRequest !== song.request;
				if (dirty) {
					if (latents) song.latents = latents;
					if (useNewRequest) {
						song.request = newRequest;
						song.caption = newRequest.caption ?? song.caption;
						song.seed = newRequest.seed ?? song.seed;
						song.duration = newRequest.duration ?? song.duration;
					}
					await putSong($state.snapshot(song));
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
		const safe = displayName.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'song';
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
		const safe = displayName.replace(/[\\/:*?"<>|\x00-\x1f]/g, '') || 'song';
		a.download = `${safe}.vae`;
		a.click();
		URL.revokeObjectURL(url);
	}

	// VAE-only scan: POST /vae with the source audio (encode path), attach
	// the fresh latents to the card non-destructively. Skips the LM
	// roundtrip. Ideal for priming a cover/repaint target without paying
	// the LM cost just to get the [VAE] badge lit.
	async function encodeOnly() {
		if (song.latents || song.id == null) return;
		scanning = true;
		try {
			const jobId = await vaeEncode(song.audio, app.request);
			saveJob('lm', jobId);
			await pollJob(jobId);
			const latents = await jobResultLatents(jobId);
			clearJob('lm');
			song.latents = latents;
			await putSong($state.snapshot(song));
		} catch (e: unknown) {
			toast(e instanceof Error ? e.message : String(e));
		} finally {
			scanning = false;
		}
	}

	let confirmDeleteOpen = $state(false);
	let confirmDeleteNonFavOpen = $state(false);
	let renameOpen = $state(false);
	let renameValue = $state('');

	async function toggleFavorite() {
		if (song.id == null) return;
		song.favorite = !song.favorite;
		await putSong($state.snapshot(song));
	}

	function openRename() {
		renameValue = song.name;
		renameOpen = true;
	}

	// Persists the new base name. The "(variant task)" suffix shown in the
	// header keeps deriving from request via displaySongName, so renaming
	// only touches the stored stem.
	async function doRename() {
		if (song.id == null) return;
		const v = renameValue.trim();
		if (!v || v === song.name) return;
		song.name = v;
		await putSong($state.snapshot(song));
	}

	async function doRemove() {
		if (song.id == null) return;
		if (app.refSongId === song.id) app.refSongId = null;
		if (app.srcSongId === song.id) app.srcSongId = null;
		await deleteSong(song.id);
		const idx = app.songs.findIndex((s) => s.id === song.id);
		if (idx >= 0) app.songs.splice(idx, 1);
	}

	// Deletes every non-favorite track in the list, regardless of which
	// card the menu was opened from. The current card is included in the
	// purge if it is not flagged favorite.
	async function doRemoveNonFavorites() {
		const victims = app.songs.filter((s) => !s.favorite);
		for (const s of victims) {
			if (s.id == null) continue;
			if (app.refSongId === s.id) app.refSongId = null;
			if (app.srcSongId === s.id) app.srcSongId = null;
			await deleteSong(s.id);
		}
		app.songs = app.songs.filter((s) => s.favorite);
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
	// flow (tweak prompt -> rename -> grab audio -> work with latents -> inspect -> destroy).
	// Destructive entries open a confirm dialog.
	const actionItems: MenuItem[] = $derived([
		{ icon: Pencil, label: 'Edit prompt', onSelect: load },
		{ icon: Type, label: 'Rename song', onSelect: openRename },
		{ icon: Download, label: 'Download audio', onSelect: downloadAudio },
		{ icon: Cpu, label: 'Compute VAE latents', onSelect: encodeOnly, disabled: !!song.latents },
		{
			icon: Download,
			label: 'Download VAE latents',
			onSelect: downloadLatents,
			disabled: !song.latents
		},
		{ icon: Ear, label: 'LM understand', onSelect: scan },
		{ icon: Trash2, label: 'Delete this track', onSelect: () => (confirmDeleteOpen = true) },
		{
			icon: TriangleAlert,
			label: 'Delete non-favorites',
			onSelect: () => (confirmDeleteNonFavOpen = true)
		}
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
		<span class="card-name">{displayName}</span>
		<Menu items={actionItems}>
			{#snippet trigger()}<ChevronDown size={14} /> Menu{/snippet}
		</Menu>
		<button
			class="icon-btn"
			onclick={toggleFavorite}
			title={song.favorite ? 'Unfavorite' : 'Favorite'}
		>
			<Heart size={14} fill={song.favorite ? 'currentColor' : 'none'} />
		</button>
	</div>
	<Waveform
		{song}
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

<Dialog bind:open={confirmDeleteOpen} title="Delete this track?" onConfirm={doRemove} />

<Dialog
	bind:open={confirmDeleteNonFavOpen}
	title="Delete non-favorites?"
	onConfirm={doRemoveNonFavorites}
/>

<Dialog bind:open={renameOpen} title="Rename song" onConfirm={doRename}>
	{#snippet body()}
		<input type="text" class="rename-input" bind:value={renameValue} />
	{/snippet}
</Dialog>

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
	.rename-input {
		width: 100%;
		background: var(--bg-input);
		border: none;
		border-radius: 3px;
		padding: 0.25rem 0.4rem;
		color: var(--fg);
		font-size: 0.8rem;
	}
	.rename-input:focus {
		outline: 1px solid var(--focus);
	}
</style>

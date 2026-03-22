<script lang="ts">
	import { Play, Square, Pencil, Download, Trash2 } from '@lucide/svelte';
	import { app } from '../lib/state.svelte.js';
	import { deleteSong } from '../lib/db.js';
	import type { Song } from '../lib/types.js';
	import Waveform from './Waveform.svelte';

	let { song }: { song: Song } = $props();

	let playing = $state(false);
	let time = $state(0);
	let dur = $state(0);

	function toggle() {
		playing = !playing;
	}

	function load() {
		app.name = song.name;
		app.request = { ...song.request };
	}

	function downloadAudio() {
		const url = URL.createObjectURL(song.audio);
		const a = document.createElement('a');
		a.href = url;
		const safe = song.name.replace(/[^a-zA-Z0-9 _-]/g, '') || 'song';
		const ext = song.format === 'wav' ? '.wav' : '.mp3';
		a.download = `${safe}_${song.seed}${ext}`;
		a.click();
		URL.revokeObjectURL(url);
	}

	async function remove() {
		if (song.id == null) return;
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
		<span class="format-badge">{song.format.toUpperCase()}</span>
		<span class="timecode">{fmtPos(time)} / {fmtDur(dur)}</span>
		<div class="card-actions">
			<button class="icon-btn" onclick={load} title="Edit"><Pencil size={14} /></button>
			<button class="icon-btn" onclick={downloadAudio} title="Download"
				><Download size={14} /></button
			>
			<button class="icon-btn" onclick={remove} title="Delete"><Trash2 size={14} /></button>
		</div>
	</div>
	<Waveform audio={song.audio} bind:playing bind:time bind:dur />
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		padding: 0.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-card);
	}
	.card-header {
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
		background: var(--border);
		color: var(--fg);
		flex-shrink: 0;
	}
	.timecode {
		font-size: 0.7rem;
		font-family: monospace;
		color: var(--fg);
		white-space: nowrap;
	}
	.card-actions {
		display: flex;
		gap: 0.2rem;
		flex-shrink: 0;
	}
	.icon-btn {
		background: none;
		border: none;
		cursor: pointer;
		padding: 0.15rem;
		color: var(--fg);
		display: flex;
		align-items: center;
	}
	.icon-btn:hover {
		color: var(--focus);
	}
</style>

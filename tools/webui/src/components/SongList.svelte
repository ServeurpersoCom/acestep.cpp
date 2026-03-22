<script lang="ts">
	import { app } from '../lib/state.svelte.js';
	import SongCard from './SongCard.svelte';
</script>

<div class="song-list">
	{#if app.songs.length === 0}
		<p class="empty">No songs yet.</p>
	{:else}
		{#each app.songs as song (song.id)}
			<SongCard {song} />
		{/each}
	{/if}
</div>

<style>
	.song-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		overflow-y: auto;
	}
	.empty {
		color: var(--fg-dim);
		font-size: 0.85rem;
		text-align: center;
		padding: 2rem 0;
	}
</style>

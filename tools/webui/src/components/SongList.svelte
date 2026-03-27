<script lang="ts">
	import { app } from '../lib/state.svelte.js';
	import SongCard from './SongCard.svelte';
	import LogCard from './LogCard.svelte';
</script>

<div class="song-list">
	{#each app.songs as song (song.id)}
		<SongCard {song} />
	{/each}
	<LogCard />
</div>

<style>
	.song-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		overflow-y: auto;
	}
</style>

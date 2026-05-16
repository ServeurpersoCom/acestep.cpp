<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		onclick,
		children
	}: {
		onclick: (e: MouseEvent) => void;
		children: Snippet;
	} = $props();
</script>

<button type="button" {onclick}>{@render children()}</button>

<style>
	button {
		background: var(--bg-btn);
		border: none;
		cursor: pointer;
		padding: 0.2rem 0.6rem;
		color: var(--fg);
		font-size: 0.8rem;
		border-radius: 3px;
	}
	button:hover {
		background: var(--bg-btn-hover);
	}
</style>

<script lang="ts">
	import { app } from '../lib/state.svelte.js';
	import { ChevronDown, ChevronRight } from '@lucide/svelte';

	let lines = $state<string[]>([]);
	let container = $state<HTMLPreElement>();

	$effect(() => {
		const es = new EventSource('logs');
		es.onmessage = (e: MessageEvent) => {
			lines.push(e.data);
			if (lines.length > 500) lines.splice(0, lines.length - 500);
			const el = container;
			if (app.logsOpen && el) {
				requestAnimationFrame(() => {
					el.scrollTop = el.scrollHeight;
				});
			}
		};
		return () => es.close();
	});
</script>

<div class="card">
	<button class="card-header" onclick={() => (app.logsOpen = !app.logsOpen)}>
		{#if app.logsOpen}
			<ChevronDown size={14} />
		{:else}
			<ChevronRight size={14} />
		{/if}
		<span class="card-label">Logs</span>
	</button>
	{#if app.logsOpen}
		<pre class="log-body" bind:this={container}>{lines.join('\n')}</pre>
	{/if}
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-card);
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.5rem;
		background: var(--bg-input);
		border: none;
		cursor: pointer;
		color: var(--fg);
		font-size: 0.8rem;
		text-align: left;
	}
	.card-label {
		font-weight: 600;
	}
	.log-body {
		margin: 0;
		padding: 0.4rem 0.5rem;
		font-family: monospace;
		font-size: 0.7rem;
		line-height: 1.4;
		color: var(--fg-dim);
		background: var(--bg-card);
		border-top: 1px solid var(--border);
		border-radius: 0 0 4px 4px;
		overflow-y: auto;
		height: 400px;
		white-space: pre-wrap;
		word-break: break-all;
	}
</style>

<script lang="ts" module>
	import type { Component } from 'svelte';

	// Public contract: one entry per row in the dropdown. Callers build the
	// array with $derived so disabled flags stay reactive to upstream state.
	// icon is any component that accepts a numeric size prop (Lucide icons
	// match this shape without binding us to @lucide/svelte here).
	export interface MenuItem {
		label: string;
		onSelect: () => void;
		disabled?: boolean;
		icon?: Component<{ size?: number }>;
	}
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		trigger,
		items,
		disabled = false
	}: { trigger: Snippet; items: MenuItem[]; disabled?: boolean } = $props();

	let open = $state(false);
	let root: HTMLDivElement;

	function toggle() {
		open = !open;
	}

	function select(item: MenuItem) {
		if (item.disabled) return;
		open = false;
		item.onSelect();
	}

	// Close on click outside and on Escape. Listeners attach only while the
	// menu is open so idle cards add zero global event cost. mousedown wins
	// over click: it fires before the item's onclick, but the check lives
	// inside root.contains so clicking an item still lets its handler run.
	$effect(() => {
		if (!open) return;
		const onDocMouseDown = (e: MouseEvent) => {
			if (!root.contains(e.target as Node)) open = false;
		};
		const onKey = (e: KeyboardEvent) => {
			if (e.key === 'Escape') open = false;
		};
		document.addEventListener('mousedown', onDocMouseDown);
		document.addEventListener('keydown', onKey);
		return () => {
			document.removeEventListener('mousedown', onDocMouseDown);
			document.removeEventListener('keydown', onKey);
		};
	});
</script>

<div class="menu" bind:this={root}>
	<button type="button" class="menu-trigger" {disabled} onclick={toggle}>
		{@render trigger()}
	</button>
	{#if open}
		<div class="menu-items">
			{#each items as item}
				{@const Icon = item.icon}
				<button
					type="button"
					class="menu-item"
					disabled={item.disabled}
					onclick={() => select(item)}
				>
					{#if Icon}<Icon size={14} />{/if}
					{item.label}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.menu {
		position: relative;
		display: inline-block;
	}
	.menu-trigger {
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
	.menu-trigger:hover {
		color: var(--focus);
	}
	.menu-items {
		position: absolute;
		top: calc(100% + 2px);
		right: 0;
		background: var(--bg-card);
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
		border-radius: 3px;
		display: flex;
		flex-direction: column;
		min-width: 8rem;
		z-index: 10;
	}
	.menu-item {
		background: none;
		border: none;
		cursor: pointer;
		padding: 0.15rem 0.5rem;
		color: var(--fg);
		text-align: left;
		font-size: 0.8rem;
		white-space: nowrap;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.menu-item:hover:not(:disabled) {
		background: var(--bg-btn-hover);
	}
	.menu-item:disabled {
		color: var(--fg-dim);
	}
</style>

<script lang="ts">
	import { app } from '../lib/state.svelte.js';
</script>

{#if app.toast}
	<div class="toast">{app.toast}</div>
{/if}

<style>
	.toast {
		position: fixed;
		bottom: 1.5rem;
		left: 50%;
		transform: translateX(-50%);
		background: var(--error);
		color: #fff;
		font-size: 0.85rem;
		padding: 0.5rem 1.2rem;
		border-radius: 6px;
		z-index: 9999;
		pointer-events: none;
		animation: fade-in 0.15s ease-out;
	}
	@keyframes fade-in {
		from {
			opacity: 0;
			transform: translateX(-50%) translateY(0.5rem);
		}
		to {
			opacity: 1;
			transform: translateX(-50%) translateY(0);
		}
	}
</style>

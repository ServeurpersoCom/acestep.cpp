<script lang="ts">
	import { onMount } from 'svelte';
	import { untrack } from 'svelte';
	import { WAVEFORM_HEIGHT } from '../lib/config.js';
	import { app } from '../lib/state.svelte.js';

	let {
		audio,
		playing = $bindable(false),
		time = $bindable(0),
		dur = $bindable(0)
	}: { audio: Blob; playing: boolean; time: number; dur: number } = $props();

	let canvas: HTMLCanvasElement;
	let peaks: number[] = [];
	let player: HTMLAudioElement | null = null;
	let url: string | null = null;
	let raf = 0;
	let dragging = false;

	// cached at setup for draw()
	let colorDim = '';
	let colorPlay = '';
	let cw = 0;
	let ch = 0;

	// one-time setup: decode audio blob, compute peaks, create player element.
	// onMount runs once and never re-triggers on prop/state changes.
	onMount(() => {
		cw = canvas.clientWidth || 300;
		ch = WAVEFORM_HEIGHT;
		canvas.width = cw;
		canvas.height = ch;

		// cache CSS colors for draw()
		const style = getComputedStyle(canvas);
		colorDim = style.getPropertyValue('--waveform-dim').trim() || '#555';
		colorPlay = style.getPropertyValue('--waveform-play').trim() || '#2ed573';

		url = URL.createObjectURL(audio);
		player = new Audio(url);
		player.volume = untrack(() => app.volume);
		player.addEventListener('ended', () => {
			playing = false;
			time = 0;
			draw(0);
		});

		const actx = new AudioContext();
		audio
			.arrayBuffer()
			.then((buf) => actx.decodeAudioData(buf))
			.then((decoded) => {
				dur = decoded.duration;
				peaks = computePeaks(decoded, cw);
				draw(0);
				actx.close();
			})
			.catch(() => {});

		return () => {
			if (player) {
				player.pause();
				player = null;
			}
			if (url) {
				URL.revokeObjectURL(url);
				url = null;
			}
			cancelLoop();
		};
	});

	// play/pause toggle
	$effect(() => {
		if (!player) return;
		if (playing) {
			player.volume = untrack(() => app.volume);
			player.play();
			startLoop();
		} else {
			player.pause();
			cancelLoop();
		}
	});

	function computePeaks(buf: AudioBuffer, numBins: number): number[] {
		const raw = buf.getChannelData(0);
		const binSize = Math.floor(raw.length / numBins);
		const out: number[] = [];
		for (let i = 0; i < numBins; i++) {
			let max = 0;
			const start = i * binSize;
			const end = Math.min(start + binSize, raw.length);
			for (let j = start; j < end; j++) {
				const v = raw[j] < 0 ? -raw[j] : raw[j];
				if (v > max) max = v;
			}
			out.push(max);
		}
		return out;
	}

	// draw waveform with playback progress (0..1)
	function draw(progress: number) {
		if (!canvas || peaks.length === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const mid = ch / 2;
		const barW = cw / peaks.length;

		ctx.clearRect(0, 0, cw, ch);

		for (let i = 0; i < peaks.length; i++) {
			const x = i * barW;
			const barH = peaks[i] * mid * 0.9;
			ctx.fillStyle = i / peaks.length <= progress ? colorPlay : colorDim;
			ctx.fillRect(x, mid - barH, Math.max(1, barW - 0.5), barH * 2);
		}

		if (progress > 0 && progress < 1) {
			ctx.fillStyle = colorPlay;
			ctx.fillRect(progress * cw - 0.5, 0, 1, ch);
		}
	}

	// rAF loop: sync volume, update time, redraw waveform
	function tick() {
		if (!player || player.paused) return;
		player.volume = app.volume;
		const t = player.currentTime;
		time = t;
		draw(dur > 0 ? t / dur : 0);
		raf = requestAnimationFrame(tick);
	}

	function startLoop() {
		cancelLoop();
		raf = requestAnimationFrame(tick);
	}

	function cancelLoop() {
		if (raf) {
			cancelAnimationFrame(raf);
			raf = 0;
		}
	}

	// seek to x position on the canvas
	function seekTo(clientX: number) {
		if (!player || !canvas || dur <= 0) return;
		const rect = canvas.getBoundingClientRect();
		const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
		player.currentTime = x * dur;
		time = player.currentTime;
		draw(x);
	}

	function onMouseDown(e: MouseEvent) {
		dragging = true;
		seekTo(e.clientX);
		if (!playing) playing = true;
		window.addEventListener('mousemove', onMouseMove);
		window.addEventListener('mouseup', onMouseUp);
	}

	function onMouseMove(e: MouseEvent) {
		if (!dragging) return;
		seekTo(e.clientX);
	}

	function onMouseUp() {
		dragging = false;
		window.removeEventListener('mousemove', onMouseMove);
		window.removeEventListener('mouseup', onMouseUp);
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<canvas bind:this={canvas} class="waveform" onmousedown={onMouseDown}></canvas>

<style>
	.waveform {
		width: 100%;
		height: var(--waveform-h, 64px);
		cursor: pointer;
		border-radius: 2px;
	}
</style>

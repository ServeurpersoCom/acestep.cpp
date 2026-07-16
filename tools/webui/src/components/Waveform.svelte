<script lang="ts">
	import { onMount, tick as svelteTick } from 'svelte';
	import { untrack } from 'svelte';
	import { WAVEFORM_HEIGHT, WAVEFORM_BINS } from '../lib/config.js';
	import { app } from '../lib/state.svelte.js';
	import { putSong } from '../lib/db.js';
	import type { Song } from '../lib/types.js';
	import {
		getContext,
		registerPlaying,
		unregisterPlaying,
		findSyncPosition,
		playingCount
	} from '../lib/audio.js';

	let {
		song,
		playing = $bindable(false),
		time = $bindable(0),
		dur = $bindable(0),
		selectable = false,
		rangeStart = $bindable(0),
		rangeEnd = $bindable(0),
		viewStart = $bindable(0),
		viewEnd = $bindable(1),
		onZoomSelect = undefined
	}: {
		song: Song;
		playing: boolean;
		time: number;
		dur: number;
		selectable: boolean;
		rangeStart: number;
		rangeEnd: number;
		viewStart: number;
		viewEnd: number;
		onZoomSelect?: (start: number, end: number) => void;
	} = $props();

	let canvas: HTMLCanvasElement;
	let peaks: Float32Array = new Float32Array();
	let displayPeaks: Float32Array = new Float32Array();
	let raf = 0;
	let cw = 0;
	let ch = 0;

	// Web Audio API
	let actx: AudioContext | null = null;
	let gain: GainNode | null = null;
	let decoded: AudioBuffer | null = null;
	let source: AudioBufferSourceNode | null = null;
	let playAt = 0;
	let playOffset = 0;
	let playingId = -1;

	// pointer state
	let dragging = false;
	let dragEdge: 'lo' | 'hi' | 'new' = 'new';
	let anchor = -1;
	let mouseOver = false;

	// shift+drag rubber-band zoom state
	let rbDragging = false;
	let rbAnchor = 0;
	let rbCurrent = 0;

	onMount(() => {
		cw = WAVEFORM_BINS;
		ch = WAVEFORM_HEIGHT;
		canvas.width = cw;
		canvas.height = ch;

		actx = getContext();
		gain = actx.createGain();
		gain.gain.value = untrack(() => app.volume);
		gain.connect(actx.destination);

		// peaks cache hit: skip decode entirely, draw from cached array
		if (song.peaks) {
			peaks = song.peaks;
			dur = song.duration;
			updateDisplayPeaks();
			draw();
		}

		song.audio
			.arrayBuffer()
			.then((buf) => actx!.decodeAudioData(buf))
			.then((buf) => {
				decoded = buf;
				dur = buf.duration;
				if (!song.peaks) {
					song.peaks = computePeaks(buf, WAVEFORM_BINS);
					if (song.id != null) putSong($state.snapshot(song));
					peaks = song.peaks;
				}
				updateDisplayPeaks();
				draw();
			})
			.catch(() => {});

		function onTouchStart(e: TouchEvent) {
			if (e.touches.length >= 2 && onZoomSelect) {
				e.preventDefault();
				const t0 = e.touches[0];
				const t1 = e.touches[1];
				rbAnchor = xToNorm(Math.min(t0.clientX, t1.clientX));
				rbCurrent = xToNorm(Math.max(t0.clientX, t1.clientX));
				rbDragging = true;
				draw();
			} else {
				e.preventDefault();
			}
		}
		function onTouchMove(e: TouchEvent) {
			if (rbDragging && e.touches.length >= 2) {
				e.preventDefault();
				const t0 = e.touches[0];
				const t1 = e.touches[1];
				rbAnchor = xToNorm(Math.min(t0.clientX, t1.clientX));
				rbCurrent = xToNorm(Math.max(t0.clientX, t1.clientX));
				draw();
			} else {
				e.preventDefault();
			}
		}
		function onTouchEnd(e: TouchEvent) {
			if (rbDragging && e.touches.length < 2) {
				rbDragging = false;
				const lo = Math.min(rbAnchor, rbCurrent);
				const hi = Math.max(rbAnchor, rbCurrent);
				if (hi - lo > (viewEnd - viewStart) * 0.02) onZoomSelect?.(lo, hi);
				draw();
			}
		}
		canvas.addEventListener('touchstart', onTouchStart, { passive: false });
		canvas.addEventListener('touchmove', onTouchMove, { passive: false });
		canvas.addEventListener('touchend', onTouchEnd);
		canvas.addEventListener('touchcancel', onTouchEnd);

		function onKeyDown(e: KeyboardEvent) {
			if (e.key === 'Shift' && onZoomSelect) canvas.style.cursor = 'crosshair';
			if (!mouseOver || viewStart === 0 && viewEnd === 1) return;
			const span = viewEnd - viewStart;
			const step = span * 0.1;
			if (e.key === 'ArrowLeft') {
				e.preventDefault();
				const ns = Math.max(0, viewStart - step);
				viewStart = ns;
				viewEnd = Math.min(1, ns + span);
				updateDisplayPeaks();
				draw();
			} else if (e.key === 'ArrowRight') {
				e.preventDefault();
				const ne = Math.min(1, viewEnd + step);
				viewEnd = ne;
				viewStart = Math.max(0, ne - span);
				updateDisplayPeaks();
				draw();
			}
		}
		function onKeyUp(e: KeyboardEvent) {
			if (e.key === 'Shift') canvas.style.cursor = '';
		}
		function onWheel(e: WheelEvent) {
			if (viewStart === 0 && viewEnd === 1) return;
			e.preventDefault();
			const raw = e.deltaX !== 0 ? e.deltaX : (e.shiftKey ? e.deltaY : 0);
			if (raw === 0) return;
			const span = viewEnd - viewStart;
			const pan = (raw / 500) * span;
			let ns = viewStart + pan;
			let ne = viewEnd + pan;
			if (ns < 0) {
				ns = 0;
				ne = span;
			}
			if (ne > 1) {
				ne = 1;
				ns = 1 - span;
			}
			viewStart = ns;
			viewEnd = ne;
			updateDisplayPeaks();
			draw();
		}
		canvas.addEventListener('wheel', onWheel, { passive: false });
		window.addEventListener('keydown', onKeyDown);
		window.addEventListener('keyup', onKeyUp);

		return () => {
			stopPlayback();
			cancelLoop();
			canvas.removeEventListener('touchstart', onTouchStart);
			canvas.removeEventListener('touchmove', onTouchMove);
			canvas.removeEventListener('touchend', onTouchEnd);
			canvas.removeEventListener('touchcancel', onTouchEnd);
			canvas.removeEventListener('wheel', onWheel);
			window.removeEventListener('keydown', onKeyDown);
			window.removeEventListener('keyup', onKeyUp);
		};
	});

	function preventTouch(e: TouchEvent) {
		e.preventDefault();
	}

	// redraw when theme, selectable, or zoom changes
	$effect(() => {
		app.dark;
		selectable;
		viewStart;
		viewEnd;
		svelteTick().then(() => {
			updateDisplayPeaks();
			if (displayPeaks.length > 0) draw();
		});
	});

	// redraw when range changes from external source (field input)
	$effect(() => {
		rangeStart;
		rangeEnd;
		if (peaks.length > 0) draw();
	});

	// play/pause
	$effect(() => {
		const wantPlay = playing;
		if (!decoded || !actx) return;
		if (wantPlay) {
			if (actx.state === 'suspended') actx.resume();
			const syncPos = findSyncPosition(dur, playingId);
			if (syncPos >= 0) {
				// schedule start 10ms in the future, compensate offset so both
				// tracks play the same sample at the same audio frame
				const now = actx.currentTime;
				const delta = 0.01;
				startPlayback(syncPos + delta, now + delta);
			} else {
				startPlayback(untrack(() => time));
			}
			startLoop();
		} else {
			stopPlayback();
			cancelLoop();
		}
	});

	function computePeaks(buf: AudioBuffer, numBins: number): Float32Array {
		const raw = buf.getChannelData(0);
		const binSize = Math.floor(raw.length / numBins);
		const out = new Float32Array(numBins);
		for (let i = 0; i < numBins; i++) {
			let max = 0;
			const start = i * binSize;
			const end = Math.min(start + binSize, raw.length);
			for (let j = start; j < end; j++) {
				const v = raw[j] < 0 ? -raw[j] : raw[j];
				if (v > max) max = v;
			}
			out[i] = max;
		}
		return out;
	}

	// re-sample raw audio for the visible window at full WAVEFORM_BINS resolution,
	// giving true higher precision when zoomed in.
	function computePeaksRange(buf: AudioBuffer, startFrac: number, endFrac: number): Float32Array {
		const raw = buf.getChannelData(0);
		const total = raw.length;
		const startSample = Math.floor(startFrac * total);
		const endSample = Math.ceil(endFrac * total);
		const rangeLen = Math.max(1, endSample - startSample);
		const out = new Float32Array(WAVEFORM_BINS);
		for (let i = 0; i < WAVEFORM_BINS; i++) {
			const s = Math.max(0, startSample + Math.floor((i / WAVEFORM_BINS) * rangeLen));
			const e = Math.min(total, startSample + Math.ceil(((i + 1) / WAVEFORM_BINS) * rangeLen));
			let max = 0;
			for (let j = s; j < Math.max(e, s + 1); j++) {
				const v = raw[j] < 0 ? -raw[j] : raw[j];
				if (v > max) max = v;
			}
			out[i] = max;
		}
		return out;
	}

	// linear interpolation over the existing peaks array when decoded audio
	// isn't yet available (cache-hit path before decode completes).
	function interpolatePeaks(src: Float32Array, startFrac: number, endFrac: number): Float32Array {
		const n = src.length;
		const out = new Float32Array(WAVEFORM_BINS);
		for (let i = 0; i < WAVEFORM_BINS; i++) {
			const pos = startFrac * n + (i / WAVEFORM_BINS) * (endFrac - startFrac) * n;
			const lo = Math.max(0, Math.min(n - 1, Math.floor(pos)));
			const hi = Math.min(n - 1, lo + 1);
			const t = pos - lo;
			out[i] = src[lo] * (1 - t) + src[hi] * t;
		}
		return out;
	}

	function updateDisplayPeaks() {
		if (peaks.length === 0) {
			displayPeaks = peaks;
			return;
		}
		if (viewStart === 0 && viewEnd === 1) {
			displayPeaks = peaks;
			return;
		}
		if (decoded) {
			displayPeaks = computePeaksRange(decoded, viewStart, viewEnd);
		} else {
			displayPeaks = interpolatePeaks(peaks, viewStart, viewEnd);
		}
	}

	function currentTime(): number {
		if (!actx || !source) return time;
		return playOffset + (actx.currentTime - playAt);
	}

	function startPlayback(offset: number, when?: number) {
		stopPlayback();
		if (!actx || !decoded || !gain) return;
		const s = actx.createBufferSource();
		s.buffer = decoded;
		s.connect(gain);
		s.onended = () => {
			if (source === s) {
				source = null;
				playing = false;
				time = 0;
				if (playingId >= 0) {
					unregisterPlaying(playingId);
					playingId = -1;
				}
				draw();
			}
		};
		playOffset = offset;
		if (when != null) {
			playAt = when;
			s.start(when, offset);
		} else {
			playAt = actx.currentTime;
			s.start(0, offset);
		}
		source = s;
		playingId = registerPlaying(dur, currentTime);
	}

	function stopPlayback() {
		if (playingId >= 0) {
			unregisterPlaying(playingId);
			playingId = -1;
		}
		if (source) {
			source.onended = null;
			try {
				source.stop();
			} catch {}
			source = null;
		}
	}

	// draw: red (range) > green (played) > gray
	// cursors: red at range edges, green at playhead (on top)
	function draw() {
		if (!canvas || displayPeaks.length === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const style = getComputedStyle(canvas);
		const colorDim = style.getPropertyValue('--waveform-dim').trim() || '#555';
		const colorPlay = style.getPropertyValue('--waveform-play').trim() || '#2ed573';
		const colorRange = style.getPropertyValue('--waveform-range').trim() || '#ff6b6b';

		const progress = dur > 0 ? currentTime() / dur : 0;
		const mid = ch / 2;
		const hasRange = rangeEnd > rangeStart && dur > 0;
		const rA = hasRange ? Math.max(0, rangeStart / dur) : 0;
		const rB = hasRange ? Math.min(1, rangeEnd / dur) : 0;

		const vs = viewStart;
		const ve = viewEnd;
		const span = Math.max(ve - vs, 1e-6);
		const n = displayPeaks.length;
		const barW = cw / n;

		ctx.clearRect(0, 0, cw, ch);

		for (let i = 0; i < n; i++) {
			const fullFrac = vs + (i / n) * span;
			const x = i * barW;
			const barH = displayPeaks[i] * mid * 0.9;
			if (hasRange && fullFrac >= rA && fullFrac < rB) {
				ctx.fillStyle = colorRange;
			} else if (fullFrac <= progress) {
				ctx.fillStyle = colorPlay;
			} else {
				ctx.fillStyle = colorDim;
			}
			ctx.fillRect(x, mid - barH, Math.max(1, barW - 0.5), barH * 2);
		}

		if (hasRange) {
			ctx.fillStyle = colorRange;
			ctx.fillRect(((rA - vs) / span) * cw - 2, 0, 4, ch);
			ctx.fillRect(((rB - vs) / span) * cw - 2, 0, 4, ch);
		}

		if (progress > 0 && progress < 1) {
			ctx.fillStyle = colorPlay;
			ctx.fillRect(((progress - vs) / span) * cw - 2, 0, 4, ch);
		}

		if (rbDragging) {
			const lo = ((Math.min(rbAnchor, rbCurrent) - vs) / span) * cw;
			const hi = ((Math.max(rbAnchor, rbCurrent) - vs) / span) * cw;
			const w = hi - lo;
			ctx.fillStyle = app.dark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)';
			ctx.fillRect(lo, 0, w, ch);
			ctx.strokeStyle = app.dark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.55)';
			ctx.lineWidth = 1.5;
			ctx.strokeRect(lo + 0.75, 0.75, Math.max(0, w - 1.5), ch - 1.5);
		}
	}

	// loop logic lives here: when playhead passes rangeEnd, restart at rangeStart
	function tick() {
		if (!source) return;
		if (gain) gain.gain.value = app.volume / Math.sqrt(playingCount());
		const t = currentTime();
		if (t >= dur) {
			stopPlayback();
			playing = false;
			time = 0;
			draw();
			return;
		}
		if (rangeEnd > rangeStart && t >= rangeEnd) {
			startPlayback(Math.max(0, rangeStart));
		}
		time = currentTime();
		draw();
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

	function xToNorm(clientX: number): number {
		if (!canvas) return 0;
		const rect = canvas.getBoundingClientRect();
		const localFrac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
		return viewStart + localFrac * (viewEnd - viewStart);
	}

	function seekTo(norm: number) {
		if (dur <= 0) return;
		time = norm * dur;
		if (source) startPlayback(time);
		draw();
	}

	function onPointerDown(e: PointerEvent) {
		canvas.setPointerCapture(e.pointerId);
		if (e.shiftKey && onZoomSelect) {
			rbDragging = true;
			rbAnchor = xToNorm(e.clientX);
			rbCurrent = rbAnchor;
			draw();
			return;
		}
		dragging = true;
		if (selectable) {
			const pos = xToNorm(e.clientX);
			const cur = pos * dur;
			// snap closest edge to mouse, or start new range
			if (rangeEnd > rangeStart && dur > 0) {
				const dLo = Math.abs(pos - rangeStart / dur);
				const dHi = Math.abs(pos - rangeEnd / dur);
				if (dLo <= dHi) {
					dragEdge = 'lo';
					rangeStart = cur;
				} else {
					dragEdge = 'hi';
					rangeEnd = cur;
				}
			} else {
				dragEdge = 'new';
				anchor = pos;
				rangeStart = cur;
				rangeEnd = cur;
			}
			draw();
		} else {
			seekTo(xToNorm(e.clientX));
		}
	}

	function onPointerMove(e: PointerEvent) {
		if (rbDragging) {
			rbCurrent = xToNorm(e.clientX);
			draw();
			return;
		}
		if (!dragging) return;
		if (selectable) {
			const cur = xToNorm(e.clientX) * dur;
			if (dragEdge === 'lo') {
				rangeStart = cur;
			} else if (dragEdge === 'hi') {
				rangeEnd = cur;
			} else {
				rangeStart = Math.min(anchor * dur, cur);
				rangeEnd = Math.max(anchor * dur, cur);
			}
			// swap edges when crossing
			if (rangeStart > rangeEnd) {
				const tmp = rangeStart;
				rangeStart = rangeEnd;
				rangeEnd = tmp;
				dragEdge = dragEdge === 'lo' ? 'hi' : 'lo';
			}
			draw();
		} else {
			seekTo(xToNorm(e.clientX));
		}
	}

	function onPointerUp() {
		if (rbDragging) {
			rbDragging = false;
			const lo = Math.min(rbAnchor, rbCurrent);
			const hi = Math.max(rbAnchor, rbCurrent);
			if (hi - lo > (viewEnd - viewStart) * 0.02) onZoomSelect?.(lo, hi);
			draw();
			return;
		}
		dragging = false;
	}

	// scrollbar pan state
	let scrollDragging = false;
	let scrollDragX = 0;
	let scrollDragViewStart = 0;

	function onScrollDown(e: PointerEvent) {
		const track = e.currentTarget as HTMLElement;
		track.setPointerCapture(e.pointerId);
		const rect = track.getBoundingClientRect();
		const clickFrac = (e.clientX - rect.left) / rect.width;
		const span = viewEnd - viewStart;
		if (clickFrac >= viewStart && clickFrac <= viewEnd) {
			scrollDragging = true;
			scrollDragX = e.clientX;
			scrollDragViewStart = viewStart;
		} else {
			// jump: center view at click
			let ns = clickFrac - span / 2;
			let ne = ns + span;
			if (ns < 0) {
				ns = 0;
				ne = span;
			}
			if (ne > 1) {
				ne = 1;
				ns = 1 - span;
			}
			viewStart = ns;
			viewEnd = ne;
			updateDisplayPeaks();
			draw();
		}
	}

	function onScrollMove(e: PointerEvent) {
		if (!scrollDragging) return;
		const track = e.currentTarget as HTMLElement;
		const rect = track.getBoundingClientRect();
		const delta = (e.clientX - scrollDragX) / rect.width;
		const span = viewEnd - viewStart;
		let ns = scrollDragViewStart + delta;
		let ne = ns + span;
		if (ns < 0) {
			ns = 0;
			ne = span;
		}
		if (ne > 1) {
			ne = 1;
			ns = 1 - span;
		}
		viewStart = ns;
		viewEnd = ne;
		updateDisplayPeaks();
		draw();
	}

	function onScrollUp() {
		scrollDragging = false;
	}
</script>

<div class="waveform-wrap">
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<canvas
		bind:this={canvas}
		class="waveform"
		onpointerdown={onPointerDown}
		onpointermove={onPointerMove}
		onpointerup={onPointerUp}
		onmouseenter={() => mouseOver = true}
		onmouseleave={() => mouseOver = false}
	></canvas>
	{#if viewStart > 0 || viewEnd < 1}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="scroll-track"
			onpointerdown={onScrollDown}
			onpointermove={onScrollMove}
			onpointerup={onScrollUp}
		>
			<div
				class="scroll-thumb"
				style="left: {viewStart * 100}%; width: {(viewEnd - viewStart) * 100}%"
			></div>
		</div>
	{/if}
</div>

<style>
	.waveform-wrap {
		display: flex;
		flex-direction: column;
		gap: 3px;
	}
	.waveform {
		width: 100%;
		height: var(--waveform-h, 64px);
		cursor: pointer;
		border-radius: 2px;
		touch-action: none;
		user-select: none;
		-webkit-user-select: none;
	}
	.scroll-track {
		height: 5px;
		background: rgba(128, 128, 128, 0.2);
		border-radius: 3px;
		position: relative;
		cursor: pointer;
		touch-action: none;
		user-select: none;
		-webkit-user-select: none;
	}
	.scroll-thumb {
		position: absolute;
		top: 0;
		height: 100%;
		background: var(--fg);
		border-radius: 3px;
		opacity: 0.55;
		pointer-events: none;
	}
</style>

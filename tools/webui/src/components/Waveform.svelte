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
		rangeEnd = $bindable(0)
	}: {
		song: Song;
		playing: boolean;
		time: number;
		dur: number;
		selectable: boolean;
		rangeStart: number;
		rangeEnd: number;
	} = $props();

	let canvas: HTMLCanvasElement;
	let peaks: Float32Array = new Float32Array();
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
					draw();
				}
			})
			.catch(() => {});

		canvas.addEventListener('touchstart', preventTouch, { passive: false });
		canvas.addEventListener('touchmove', preventTouch, { passive: false });

		return () => {
			stopPlayback();
			cancelLoop();
			canvas.removeEventListener('touchstart', preventTouch);
			canvas.removeEventListener('touchmove', preventTouch);
		};
	});

	function preventTouch(e: TouchEvent) {
		e.preventDefault();
	}

	// redraw when theme or selectable state changes
	$effect(() => {
		app.dark;
		selectable;
		svelteTick().then(() => {
			if (peaks.length > 0) draw();
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
		if (!canvas || peaks.length === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const style = getComputedStyle(canvas);
		const colorDim = style.getPropertyValue('--waveform-dim').trim() || '#555';
		const colorPlay = style.getPropertyValue('--waveform-play').trim() || '#2ed573';
		const colorRange = style.getPropertyValue('--waveform-range').trim() || '#ff6b6b';

		const progress = dur > 0 ? currentTime() / dur : 0;
		const mid = ch / 2;
		const barW = cw / peaks.length;
		const hasRange = rangeEnd > rangeStart && dur > 0;
		const rA = hasRange ? Math.max(0, rangeStart / dur) : 0;
		const rB = hasRange ? Math.min(1, rangeEnd / dur) : 0;

		ctx.clearRect(0, 0, cw, ch);

		for (let i = 0; i < peaks.length; i++) {
			const frac = i / peaks.length;
			const x = i * barW;
			const barH = peaks[i] * mid * 0.9;
			if (hasRange && frac >= rA && frac < rB) {
				ctx.fillStyle = colorRange;
			} else if (frac <= progress) {
				ctx.fillStyle = colorPlay;
			} else {
				ctx.fillStyle = colorDim;
			}
			ctx.fillRect(x, mid - barH, Math.max(1, barW - 0.5), barH * 2);
		}

		if (hasRange) {
			ctx.fillStyle = colorRange;
			ctx.fillRect(rA * cw - 2, 0, 4, ch);
			ctx.fillRect(rB * cw - 2, 0, 4, ch);
		}

		if (progress > 0 && progress < 1) {
			ctx.fillStyle = colorPlay;
			ctx.fillRect(progress * cw - 2, 0, 4, ch);
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
		return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
	}

	function seekTo(norm: number) {
		if (dur <= 0) return;
		time = norm * dur;
		if (source) startPlayback(time);
		draw();
	}

	function onPointerDown(e: PointerEvent) {
		dragging = true;
		canvas.setPointerCapture(e.pointerId);
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
		dragging = false;
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<canvas
	bind:this={canvas}
	class="waveform"
	onpointerdown={onPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
></canvas>

<style>
	.waveform {
		width: 100%;
		height: var(--waveform-h, 64px);
		cursor: pointer;
		border-radius: 2px;
		touch-action: none;
		user-select: none;
		-webkit-user-select: none;
	}
</style>

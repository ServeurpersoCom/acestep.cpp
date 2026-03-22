import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { viteSingleFile } from 'vite-plugin-singlefile';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { gzipSync } from 'zlib';
import { resolve } from 'path';

// deterministic gzip of the inlined index.html into ../public/index.html.gz.
// the .gz is committed to git so the C++ build works without npm.
// gzip timestamp and OS bytes are zeroed for reproducible output.
function aceGzipPlugin() {
	return {
		name: 'ace:gzip',
		apply: 'build' as const,
		closeBundle() {
			const indexPath = resolve(__dirname, 'dist', 'index.html');
			const publicDir = resolve(__dirname, '..', 'public');
			const gzPath = resolve(publicDir, 'index.html.gz');

			const html = readFileSync(indexPath, 'utf-8');

			const compressed = gzipSync(Buffer.from(html, 'utf-8'), { level: 9 });

			// zero gzip header fields that vary between builds
			compressed[4] = 0; // mtime
			compressed[5] = 0;
			compressed[6] = 0;
			compressed[7] = 0;
			compressed[9] = 0; // OS

			mkdirSync(publicDir, { recursive: true });
			writeFileSync(gzPath, compressed);

			console.log(
				`  index.html: ${html.length} bytes -> index.html.gz: ${compressed.length} bytes`
			);
		}
	};
}

export default defineConfig({
	plugins: [svelte(), viteSingleFile(), aceGzipPlugin()],

	// dev server: proxy ace-server endpoints
	server: {
		proxy: {
			'/lm': 'http://localhost:8080',
			'/synth': 'http://localhost:8080',
			'/understand': 'http://localhost:8080',
			'/health': 'http://localhost:8080',
			'/props': 'http://localhost:8080'
		}
	},

	build: {
		assetsInlineLimit: Infinity,
		cssCodeSplit: false
	}
});

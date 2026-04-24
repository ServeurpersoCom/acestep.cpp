import type { Song, AceRequest } from './types.js';

const DB_NAME = 'ace-songs';
const DB_VERSION = 1;
const STORE = 'songs';

// Module-scoped singleton: one IDB connection for the whole page lifetime.
// The old pattern reopened the DB on every tx(), which leaked handles and
// eventually tripped the "unrecoverable UnknownError" under concurrent
// writes. Connection is lazy, retriable on error (null reset), and shared
// by every put/get/delete call below.
let dbPromise: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
	if (dbPromise) return dbPromise;
	dbPromise = new Promise((resolve, reject) => {
		const req = indexedDB.open(DB_NAME, DB_VERSION);
		req.onupgradeneeded = () => {
			const db = req.result;
			if (!db.objectStoreNames.contains(STORE)) {
				db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
			}
		};
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => {
			dbPromise = null;
			reject(req.error);
		};
	});
	return dbPromise;
}

// wrap a single IDB transaction operation into a promise
function tx<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
	return open().then(
		(db) =>
			new Promise((resolve, reject) => {
				const store = db.transaction(STORE, mode).objectStore(STORE);
				const req = fn(store);
				req.onsuccess = () => resolve(req.result);
				req.onerror = () => reject(req.error);
			})
	);
}

export function putSong(song: Song): Promise<number> {
	// IDBValidKey -> number (autoIncrement)
	return tx('readwrite', (s) => s.put(song)) as Promise<number>;
}

export function getAllSongs(): Promise<Song[]> {
	return tx('readonly', (s) => s.getAll());
}

export function deleteSong(id: number): Promise<void> {
	return tx('readwrite', (s) => s.delete(id)) as Promise<void>;
}

// pending jobs: saved before polling starts, cleared on completion.
// stores the job ID and track metadata for proper SongCard creation.

interface PendingJob {
	id: string;
	name: string;
	format: string;
	variant: string;
	tracks: Array<{
		caption: string;
		seed: number;
		duration: number;
		task: string;
		request: AceRequest;
	}>;
}

export function saveJob(key: string, job: PendingJob | string) {
	const val = typeof job === 'string' ? job : JSON.stringify(job);
	localStorage.setItem('ace-job-' + key, val);
}

export function loadJob(key: string): PendingJob | null {
	const raw = localStorage.getItem('ace-job-' + key);
	if (!raw) return null;
	try {
		return JSON.parse(raw);
	} catch {
		return null;
	}
}

export function loadJobId(key: string): string | null {
	const raw = localStorage.getItem('ace-job-' + key);
	if (!raw) return null;
	try {
		const parsed = JSON.parse(raw);
		return parsed.id;
	} catch {
		return raw;
	}
}

export function clearJob(key: string) {
	localStorage.removeItem('ace-job-' + key);
}

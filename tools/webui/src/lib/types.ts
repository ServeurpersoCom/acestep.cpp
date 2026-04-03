// mirrors AceRequest from request.h
// all fields optional except caption: empty/unset = server applies default
export interface AceRequest {
	caption: string;
	lyrics?: string;
	audio_codes?: string;
	bpm?: number;
	duration?: number;
	keyscale?: string;
	timesignature?: string;
	vocal_language?: string;
	seed?: number;
	lm_batch_size?: number;
	synth_batch_size?: number;
	lm_temperature?: number;
	lm_cfg_scale?: number;
	lm_top_p?: number;
	lm_top_k?: number;
	lm_negative_prompt?: string;
	use_cot_caption?: boolean;
	inference_steps?: number;
	guidance_scale?: number;
	shift?: number;
	audio_cover_strength?: number;
	cover_noise_strength?: number;
	repainting_start?: number;
	repainting_end?: number;
	repaint_strength?: number;
	task_type?: string;
	track?: string;
	// server routing (not part of C++ AceRequest, parsed separately)
	synth_model?: string;
	lm_model?: string;
	lora?: string;
	lora_scale?: number;
}

// GET /props response
export interface AceProps {
	version: string;
	models: {
		lm: string[];
		embedding: string[];
		dit: string[];
		vae: string[];
	};
	loras: string[];
	cli: Record<string, string | number>;
	default: AceRequest;
}

// what we store in IndexedDB per song
export interface Song {
	id?: number;
	name: string;
	format: string;
	created: number;
	caption: string;
	seed: number;
	duration: number;
	request: AceRequest;
	audio: Blob;
}

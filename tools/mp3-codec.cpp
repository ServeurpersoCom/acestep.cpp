// mp3-codec.cpp: MP3 encoder/decoder CLI.
//
// Encode: mp3-codec -i input.wav -o output.mp3 [-b 128]
// Decode: mp3-codec -i input.mp3 -o output.wav
//
// Direction is auto-detected from output extension.
// Encoder: acestep mp3enc (MIT). Decoder: minimp3 (CC0).

#include "audio-io.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool ends_with(const char * str, const char * suffix) {
    int slen = (int) strlen(str);
    int xlen = (int) strlen(suffix);
    if (slen < xlen) {
        return false;
    }
    for (int i = 0; i < xlen; i++) {
        char a = str[slen - xlen + i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') {
            a += 32;
        }
        if (b >= 'A' && b <= 'Z') {
            b += 32;
        }
        if (a != b) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
        fprintf(stderr,
                "Usage: %s -i <input> -o <output> [options]\n"
                "\n"
                "  -i <path>          Input file (WAV or MP3)\n"
                "  -o <path>          Output file (WAV or MP3)\n"
                "  -b <kbps>          Bitrate for MP3 encoding (default: 128)\n"
                "  --wav-format <fmt> WAV audio format (default: wav16)\n"
                "                       Requires use of -o with a .wav extension\n"
                "                       Supported values: wav, wav16, wav24, wav32\n"
                "                         wav/wav16: 16-bit signed-integer PCM audio\n"
                "                         wav24: 24-bit signed-integer PCM audio\n"
                "                         wav32: 32-bit IEEE floating-point PCM audio\n"
                "                           (wav32 disables normalization & peak clip)\n"
                "\n"
                "Mode is auto-detected from output extension.\n"
                "\n"
                "Examples:\n"
                "  %s -i song.wav -o song.mp3\n"
                "  %s -i song.wav -o song.mp3 -b 192\n"
                "  %s -i song.mp3 -o song.wav\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char * input   = NULL;
    const char * output  = NULL;
    int          bitrate = 128;
    const char * wav_format_str = nullptr;
    WavFormat    wav_format     = WAV_FORMAT_S16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bitrate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--wav-format") == 0 && i + 1 < argc) {
            wav_format_str = argv[++i];
        } else {
            fprintf(stderr, "[MP3-Codec] Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (!input || !output) {
        fprintf(stderr, "[MP3-Codec] Both -i and -o are required\n");
        return 1;
    }
    if (!ends_with(output, ".wav") && wav_format_str != nullptr) {
        fprintf(stderr, "[MP3-Codec] ERROR: --wav-format requires .wav output file\n");
        return 1;
    }
    if (!parse_optional_wav_format(wav_format_str, wav_format)) {
        fprintf(stderr, "[MP3-Codec] ERROR: --wav-format requires a supported value\n");
        return 1;
    }

    // read input (WAV or MP3, auto-detected)
    int     T = 0, sr = 0;
    float * audio = audio_read(input, &T, &sr);
    if (!audio) {
        return 1;
    }

    // write output (WAV or MP3, auto-detected from extension)
    bool ok;
    if (ends_with(output, ".mp3")) {
        ok = audio_write_mp3(output, audio, T, sr, bitrate);
    } else if (ends_with(output, ".wav")) {
        ok = audio_write_wav(output, audio, T, sr, wav_format);
    } else {
        fprintf(stderr, "[MP3-Codec] Cannot determine format from output extension\n");
        fprintf(stderr, "  use .mp3 for encoding, .wav for decoding\n");
        free(audio);
        return 1;
    }

    free(audio);
    return ok ? 0 : 1;
}

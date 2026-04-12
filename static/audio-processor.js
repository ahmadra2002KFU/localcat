class MicProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = [];
        this._bufferSize = 0;
        // Target: 1600 samples at 16kHz = 100ms chunks
        // Browser runs at 48kHz, we downsample by picking every 3rd sample
        // So we need 4800 samples at 48kHz to get 1600 at 16kHz
        this._targetSamples = 4800;
    }

    process(inputs) {
        const input = inputs[0][0]; // mono
        if (!input) return true;

        this._buffer.push(new Float32Array(input));
        this._bufferSize += input.length;

        if (this._bufferSize >= this._targetSamples) {
            // Concatenate buffer
            const full = new Float32Array(this._bufferSize);
            let offset = 0;
            for (const chunk of this._buffer) {
                full.set(chunk, offset);
                offset += chunk.length;
            }

            // Downsample 48kHz -> 16kHz (pick every 3rd sample)
            const downsampled = new Float32Array(Math.floor(full.length / 3));
            for (let i = 0; i < downsampled.length; i++) {
                downsampled[i] = full[i * 3];
            }

            // Convert to int16
            const int16 = new Int16Array(downsampled.length);
            for (let i = 0; i < downsampled.length; i++) {
                const s = Math.max(-1, Math.min(1, downsampled[i]));
                int16[i] = s < 0 ? s * 32768 : s * 32767;
            }

            this.port.postMessage(int16.buffer, [int16.buffer]);
            this._buffer = [];
            this._bufferSize = 0;
        }

        return true;
    }
}

registerProcessor('mic-processor', MicProcessor);

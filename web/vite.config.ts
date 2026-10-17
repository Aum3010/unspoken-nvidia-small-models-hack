import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // ONNX Runtime Web loads its own .wasm/.mjs runtime at run time (from /ort/); pre-bundling it breaks that
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    port: 5317,
    strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})

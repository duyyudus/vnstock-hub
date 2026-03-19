import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { execSync } from 'node:child_process'

const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'))

function getBuildNumber(): string {
  // CI/Docker: set VITE_APP_BUILD as a build arg before `npm run build`
  if (process.env.VITE_APP_BUILD) return process.env.VITE_APP_BUILD
  // Local dev fallback: count commits
  try {
    return execSync('git rev-list --count HEAD', { encoding: 'utf-8' }).trim()
  } catch {
    return 'unknown'
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __APP_BUILD__: JSON.stringify(getBuildNumber()),
  },
})

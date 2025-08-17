#!/usr/bin/env bash
set -euo pipefail

# QuantumOptim SaaS - Frontend scaffolding for Vite + React + Tailwind
# This script creates the minimal files required for a Vercel build.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Scaffolding frontend files..."
mkdir -p frontend/src/pages frontend/src/services

# index.html
cat > frontend/index.html <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>QuantumOptim by AYNX AI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
EOF

# tsconfig.json
cat > frontend/tsconfig.json <<'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "ES2020", "DOM.Iterable"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": false,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
EOF

# vite.config.ts
cat > frontend/vite.config.ts <<'EOF'
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
});
EOF

# Tailwind config
cat > frontend/tailwind.config.cjs <<'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
EOF

# PostCSS config
cat > frontend/postcss.config.cjs <<'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
EOF

# index.css
cat > frontend/src/index.css <<'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji; }
EOF

# main.tsx
cat > frontend/src/main.tsx <<'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import { HelmetProvider } from 'react-helmet-async';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <HelmetProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </HelmetProvider>
  </React.StrictMode>
);
EOF

# App.tsx
cat > frontend/src/App.tsx <<'EOF'
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
    </Routes>
  );
};

export default App;
EOF

# API status service (optional for connectivity checks)
cat > frontend/src/services/api.ts <<'EOF'
export const API_URL = (import.meta as any).env?.VITE_API_URL || (import.meta as any).env?.REACT_APP_API_URL || '';

export async function getStatus(): Promise<{ status: string } | null> {
  if (!API_URL) return null;
  try {
    const res = await fetch(`${API_URL}/api/v1/status`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
EOF

chmod +x "$0" || true

echo "Frontend scaffolding complete. Files created:"
ls -la frontend || true
ls -la frontend/src || true

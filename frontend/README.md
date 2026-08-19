# FactoryMind AI frontend

React and TypeScript interface for the FactoryMind AI predictive-maintenance API.

## Local development

```bash
npm install
npm run dev
```

The frontend calls `http://127.0.0.1:8000` by default. Override it with a
`VITE_API_BASE_URL` environment variable when needed.

## Validation

```bash
npm run lint
npm test
```

The current product shell includes Overview, Machine Analysis, and Model Info.
Future modules are visibly marked Coming Soon and contain no simulated data.

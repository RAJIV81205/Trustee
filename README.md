# Trustee

Trustee is a developer-focused prototype and sandbox that combines a modern React frontend, Node.js backend services, and Aptos Move smart contracts to simulate a simple on-chain market and end-to-end trading flows. It is designed to help developers prototype wallet integrations, transaction generation, market updates, and AI-driven investment insights against a reproducible Aptos devnet environment.

Why this exists
- Local Web3 finance stacks are hard to reproduce: frontends, market feeders, and on-chain contracts are typically in separate repos and require manual wiring.
- Trustee bundles the pieces needed to prototype trading UX, transaction flows, and market updates so teams can iterate quickly and test integrations without production risk.

What problem it solves
- Provides a single repo with a UI, backend services, and Move sources so teams can:
	- Connect and test Aptos wallets (Petra) in the browser.
	- Generate and inspect Aptos transaction payloads for buy/sell flows.
	- Simulate market updates and push price changes to Move contracts.
	- Evaluate AI analysis/UIs with deterministic mock data or live price feeds.

Repository layout (high level)
- /src — Vite + React + TypeScript frontend (pages: `Index`, `Portfolio`, UI components)
- /vite.config.ts — Vite config (dev server defaults to host :: and port 8080)
- /aptos-move-project — Move sources, Node backends and scripts:
	- /sources — Move modules for mock coins and a stock market
	- `buy.js` — API server for market data, portfolio and transaction generation (port 4002)
	- `server1.js` — market updater and Aptos push logic (port 4001)
	- `server.js` — alternate market/update server (port 4003)
	- /scripts — helper scripts (`run.sh`, `deploy.sh`) that call the Aptos CLI

Key features
- Frontend: market overview, charts, buy/sell dialogs, wallet connect (Petra), portfolio view, AI insights placeholders.
- Backends: REST endpoints for `/market-data`, `/portfolio`, `/stock-portfolio`, `/buy-stock`, `/sell-stock` and utilities to generate unsigned Aptos transactions.
- Move contracts: mock coins and a stock market to simulate on-chain state for testing.
- Fallbacks: services fall back to mock/synthetic data if external APIs (CoinGecko / AlphaVantage) fail.

Prerequisites
- Node.js 18+ and npm (or your preferred package manager)
- Aptos CLI installed and configured with a profile (the Node services call `aptos move run` and expect a `default` profile in some scripts)
- (Optional) Alpha Vantage API key if you want live stock prices
- Internet access for external price APIs (CoinGecko/AlphaVantage) or use mock data

Quick start (local development)
1) Frontend
```bash
# From repo root
cd /workspaces/Trustee
npm install
# Start Vite dev server (may be `npm run dev` or `npm start` depending on package.json)
npm run dev
# Visit http://localhost:8080 (vite.config.ts sets port 8080 by default)
```

2) Backends (run in separate terminals)
```bash
cd /workspaces/Trustee/aptos-move-project
npm install

# market updater (server1.js) - default port 4001
node server1.js

# buy/transaction server (buy.js) - default port 4002
node buy.js

# alternate market server (server.js) - default port 4003
node server.js
```

3) (Optional) Publish Move modules and inspect events
```bash
cd /workspaces/Trustee/aptos-move-project
# Check and edit named addresses in scripts/deploy.sh before running
bash ./scripts/deploy.sh
# View market data via helper script
bash ./scripts/run.sh
```

Environment variables & notes
- `ALPHA_VANTAGE_API_KEY` — set this if you want live stock lookups in `server1.js`:
```bash
export ALPHA_VANTAGE_API_KEY="your_key_here"
```
- Aptos CLI must be installed and configured if you will run the provided deploy/run scripts. Many server scripts call the CLI via `execSync` with `--profile default`.
- Do not run these scripts against mainnet accounts or with keys you care about; this repo is designed for devnet/testing.

Ports used by repository (defaults found in code)
- Frontend Vite dev server: 8080
- market updater (`server1.js`): 4001
- buy/transaction server (`buy.js`): 4002
- alternate market server (`server.js`): 4003

How the pieces connect (example flow)
1. Frontend fetches `/market-data` from the local market server to render prices.
2. User connects a Petra wallet (`window.aptos`) and requests a buy/sell.
3. Frontend calls `/buy-stock` or `/sell-stock` on the `buy.js` backend. Backend generates an unsigned Aptos transaction (or executes `aptos move run` for server-side flows) and returns the payload.
4. Developer can inspect the payload, sign with the wallet, and submit the transaction to devnet.

Security & caveats
- Several Node services execute `aptos` CLI commands via shell. This requires a trusted environment. Do not expose these servers publicly without authentication.
- The repo includes mock addresses and demo keys in scripts — replace them before any non-test deployment.

Troubleshooting
- If the frontend fails to load, confirm the Vite dev server started and check the port in `vite.config.ts`.
- If market endpoints return mock data, check internet connectivity or provide `ALPHA_VANTAGE_API_KEY`.
- If `aptos` CLI commands fail, ensure the CLI is installed, the `default` profile exists and has keys configured.

Contributing and next steps
- Add tests for the backend endpoints and Move view functions.
- Optional: add a single `dev` script that starts the frontend and backends concurrently or provide a `docker-compose` for reproducible local launches.

License
- Add your chosen license here (MIT, Apache-2.0, etc.).

-- End of file --

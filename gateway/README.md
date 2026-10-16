# ImageFlow Gateway

Cloudflare Worker gateway for the desktop package.

Routes:

- `POST /v1/classify` -> Cloudflare Workers AI Clef / Clef Flash
- `POST /v1/edit` -> OpenRouter Image API -> `bytedance-seed/seedream-5-0-flash`

## Deploy

```bash
cd gateway
npm install
npx wrangler login

npx wrangler secret put IMAGEFLOW_DEVICE_TOKEN
npx wrangler secret put OPENROUTER_API_KEY

npm run deploy
```

After deploy, Wrangler prints a URL such as:

```text
https://imageflow-gateway.<your-subdomain>.workers.dev
```

Configure the desktop app:

Windows PowerShell:

```powershell
$env:IMAGEFLOW_GATEWAY_URL="https://imageflow-gateway.<your-subdomain>.workers.dev"
$env:IMAGEFLOW_DEVICE_TOKEN="<same device token>"
imageflow ui
```

The device token is your own random shared secret between the desktop client and Worker.
Do not use the OpenRouter key as the device token.

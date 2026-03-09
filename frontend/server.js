/**
 * Local dev server with API proxy.
 * Serves static files AND proxies /api/* to Modal endpoints.
 * This completely bypasses CORS since the browser only talks to localhost.
 *
 * Usage: node server.js
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = process.env.PORT || 4000;

// Modal API base URLs
const DESIGN_API = 'https://alfred-sjoqvist--boltzgen-trem2-design-web.modal.run';
const SCORE_API = 'https://alfred-sjoqvist--boltz2-trem2-api-web.modal.run';

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

function proxyRequest(targetUrl, req, res) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const parsed = url.parse(targetUrl);
      const options = {
        hostname: parsed.hostname,
        port: 443,
        path: parsed.path,
        method: req.method,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        timeout: 1800000, // 30 min timeout for long GPU jobs
      };

      console.log(`[PROXY] ${req.method} ${targetUrl} (${body.length} bytes)`);

      const proxyReq = https.request(options, (proxyRes) => {
        let data = '';
        proxyRes.on('data', chunk => data += chunk);
        proxyRes.on('end', () => {
          console.log(`[PROXY] Response: ${proxyRes.statusCode} (${data.length} bytes)`);
          res.writeHead(proxyRes.statusCode, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          });
          res.end(data);
          resolve();
        });
      });

      proxyReq.on('error', (e) => {
        console.error(`[PROXY] Error: ${e.message}`);
        res.writeHead(502, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
        res.end(JSON.stringify({ error: `Proxy error: ${e.message}` }));
        resolve();
      });

      proxyReq.on('timeout', () => {
        console.error(`[PROXY] Timeout after 30 min`);
        proxyReq.destroy();
        res.writeHead(504, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
        res.end(JSON.stringify({ error: 'Proxy timeout (30 min)' }));
        resolve();
      });

      if (body) proxyReq.write(body);
      proxyReq.end();
    });
  });
}

const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(200, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400',
    });
    res.end();
    return;
  }

  // API proxy routes
  if (pathname.startsWith('/api/design')) {
    const apiPath = pathname.replace('/api/design', '') || '/';
    await proxyRequest(DESIGN_API + apiPath, req, res);
    return;
  }
  if (pathname.startsWith('/api/score')) {
    const apiPath = pathname.replace('/api/score', '') || '/';
    await proxyRequest(SCORE_API + apiPath, req, res);
    return;
  }

  // Static file serving
  let filePath = pathname === '/' ? '/index.html' : pathname;
  filePath = path.join(__dirname, filePath);

  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  } catch (e) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`\n  TREM2 Binder Lab running at: http://localhost:${PORT}\n`);
  console.log(`  Proxy routes:`);
  console.log(`    /api/design/* -> ${DESIGN_API}/*`);
  console.log(`    /api/score/*  -> ${SCORE_API}/*`);
  console.log(`\n  All CORS issues bypassed.\n`);
});

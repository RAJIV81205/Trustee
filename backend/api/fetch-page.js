import puppeteer from "puppeteer-core";
import chromium from "chrome-aws-lambda";

export default async function handler(req, res) {
  // Handle CORS
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, message: "Method not allowed" });
  }

  const { url } = req.body;
  if (!url) {
    return res.status(400).json({ success: false, message: "URL is required" });
  }

  let browser;
  let page;
  try {
    browser = await puppeteer.launch({
      args: chromium.args,
      defaultViewport: chromium.defaultViewport,
      executablePath: await chromium.executablePath,
      headless: chromium.headless,
      ignoreHTTPSErrors: true,
    });

    page = await browser.newPage();
    
    // Set viewport to simulate a larger screen for better video support
    await page.setViewport({ width: 1920, height: 1080 });
    
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });

    const content = await page.content();
    const title = await page.title();

    res.json({ success: true, title, content, url });
  } catch (err) {
    console.error('Error fetching page:', err);
    res.status(500).json({ success: false, message: err.message });
  } finally {
    if (page) await page.close();
    if (browser) await browser.close();
  }
}
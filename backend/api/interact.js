import puppeteer from "puppeteer-core";
import chromium from "chrome-aws-lambda";

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, message: "Method not allowed" });
  }

  const { url, action, data } = req.body;
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
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });

    if (action === "click") {
      await page.click(data.selector);
    } else if (action === "type") {
      await page.type(data.selector, data.text);
    }

    const content = await page.content();
    const title = await page.title();

    res.json({ success: true, title, content, url: page.url() });
  } catch (err) {
    console.error('Error in interaction:', err);
    res.status(500).json({ success: false, message: err.message });
  } finally {
    if (page) await page.close();
    if (browser) await browser.close();
  }
}
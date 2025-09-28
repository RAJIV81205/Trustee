import express from "express";
import cors from "cors";
import puppeteer from "puppeteer-core";
import chromium from "chrome-aws-lambda";

const app = express();
const PORT = 3001; // change if you want

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Create browser instance for each request (serverless-friendly)
async function createBrowser() {
  return await puppeteer.launch({
    args: chromium.args,
    defaultViewport: chromium.defaultViewport,
    executablePath: await chromium.executablePath,
    headless: chromium.headless,
    ignoreHTTPSErrors: true,
  });
}

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", serverless: true });
});

// Fetch and display a page
app.post("/api/fetch-page", async (req, res) => {
  const { url } = req.body;
  if (!url) {
    return res.status(400).json({ success: false, message: "URL is required" });
  }

  let browser;
  let page;
  try {
    browser = await createBrowser();
    page = await browser.newPage();
    
    // Set viewport to simulate a larger screen for better video support
    await page.setViewport({ width: 1920, height: 1080 });
    
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });

    const content = await page.content();
    const title = await page.title();

    res.json({ success: true, title, content, url });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  } finally {
    if (page) await page.close();
    if (browser) await browser.close();
  }
});

// Simple interaction API
app.post("/api/interact", async (req, res) => {
  const { url, action, data } = req.body;
  if (!url) return res.status(400).json({ success: false, message: "URL is required" });

  let browser;
  let page;
  try {
    browser = await createBrowser();
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
    res.status(500).json({ success: false, message: err.message });
  } finally {
    if (page) await page.close();
    if (browser) await browser.close();
  }
});

// Export for Vercel
export default app;

// For local development
if (process.env.NODE_ENV !== 'production') {
  app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

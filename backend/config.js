// Configuration for the browser proxy server
export const config = {
    // Server settings
    port: 3001,
    
    // Browser settings
    browser: {
        headless: 'new',
        defaultViewport: {
            width: 1920,
            height: 1080,
            deviceScaleFactor: 1,
        },
        // VPN and proxy settings
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            // VPN friendly settings
            '--ignore-certificate-errors',
            '--ignore-ssl-errors',
            '--ignore-certificate-errors-spki-list',
            '--disable-extensions-except',
            '--disable-plugins-discovery',
            '--allow-running-insecure-content',
            // Additional privacy settings
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection'
        ]
    },
    
    // US location settings (for geolocation spoofing)
    location: {
        latitude: 40.7128,   // New York
        longitude: -74.0060,
        accuracy: 100
    },
    
    // User agent for US browsing
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    
    // HTTP headers to appear as US user
    headers: {
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Upgrade-Insecure-Requests': '1'
    },
    
    // Timeout settings
    timeouts: {
        navigation: 30000,
        fallback: 20000,
        simple: 15000,
        wait: 2000
    },
    
    // Sites that commonly need special handling
    specialSites: {
        streaming: [
            'netflix.com',
            'hulu.com',
            'disney.com',
            'amazon.com/prime',
            'hbo.com',
            'paramount.com'
        ],
        social: [
            'facebook.com',
            'twitter.com',
            'instagram.com',
            'tiktok.com'
        ]
    }
};

export default config;
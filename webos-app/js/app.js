class BrowserProxyApp {
    constructor() {
        this.serverIp = localStorage.getItem('serverIp') || '192.168.29.7';
        this.currentUrl = '';
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSettings();

        // Focus on URL input for TV remote navigation
        document.getElementById('urlInput').focus();
    }

    bindEvents() {
        // URL input and go button
        document.getElementById('goButton').addEventListener('click', () => this.loadWebsite());
        document.getElementById('urlInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.loadWebsite();
            }
        });

        // Settings
        document.getElementById('settingsButton').addEventListener('click', () => this.toggleSettings());
        document.getElementById('saveSettings').addEventListener('click', () => this.saveSettings());
        document.getElementById('testConnection').addEventListener('click', () => this.testConnection());

        // Error handling
        document.getElementById('retryButton').addEventListener('click', () => this.loadWebsite());

        // Handle TV remote navigation
        document.addEventListener('keydown', (e) => this.handleRemoteNavigation(e));

        // Handle fullscreen changes
        document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
        document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
        document.addEventListener('mozfullscreenchange', () => this.handleFullscreenChange());

        // Listen for fullscreen requests from iframe
        window.addEventListener('message', (event) => this.handleIframeMessage(event));
    }

    handleRemoteNavigation(e) {
        // Handle WebOS TV remote keys
        switch (e.keyCode) {
            case 13: // Enter/OK button
                const activeElement = document.activeElement;
                if (activeElement.tagName === 'BUTTON') {
                    activeElement.click();
                } else if (activeElement.id === 'urlInput') {
                    this.loadWebsite();
                }
                break;
            case 8: // Back button
                // Exit fullscreen if in fullscreen, otherwise show welcome
                if (document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement) {
                    this.exitFullscreen();
                } else {
                    this.showWelcome();
                }
                break;
            case 70: // F key for fullscreen
                if (document.querySelector('.website-frame')) {
                    this.enterFullscreenMode();
                }
                break;
        }
    }

    toggleSettings() {
        const panel = document.getElementById('settingsPanel');
        panel.classList.toggle('hidden');

        if (!panel.classList.contains('hidden')) {
            document.getElementById('serverIp').focus();
        }
    }

    loadSettings() {
        if (this.serverIp) {
            document.getElementById('serverIp').value = this.serverIp;
        }
    }

    saveSettings() {
        const serverIp = document.getElementById('serverIp').value.trim();
        if (serverIp) {
            this.serverIp = serverIp;
            localStorage.setItem('serverIp', serverIp);
            this.showStatus('Settings saved!', 'success');
            setTimeout(() => this.toggleSettings(), 1000);
        } else {
            this.showStatus('Please enter a valid IP address', 'error');
        }
    }

    async testConnection() {
        if (!this.serverIp) {
            this.showStatus('Please enter server IP first', 'error');
            return;
        }

        try {
            const response = await fetch(`http://${this.serverIp}:3001/api/health`);
            if (response.ok) {
                this.showStatus('Connection successful!', 'success');
            } else {
                this.showStatus('Server responded with error', 'error');
            }
        } catch (error) {
            this.showStatus('Connection failed', 'error');
        }
    }

    showStatus(message, type) {
        const status = document.getElementById('connectionStatus');
        status.textContent = message;
        status.className = `status ${type}`;

        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 3000);
    }

    async loadWebsite() {
        const url = document.getElementById('urlInput').value.trim();

        if (!url) {
            this.showError('Please enter a URL');
            return;
        }

        if (!this.serverIp) {
            this.showError('Please configure server IP in settings first');
            return;
        }

        this.currentUrl = url;
        this.showLoading('Loading website...');

        try {
            this.updateLoadingMessage('Loading website content...');

            const response = await fetch(`http://${this.serverIp}:3001/api/fetch-page`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url })
            });

            const data = await response.json();

            if (data.success) {
                this.displayWebsite(data.content, data.title, data.loadingStrategy);
                console.log('Website loaded successfully with strategy:', data.loadingStrategy);
            } else {
                this.showError(data.message || 'Failed to load website.');
            }
        } catch (error) {
            console.error('Fetch error:', error);
            this.showError('Unable to load website. Please check your URL and server connection.');
        }
    }

    displayWebsite(content, title, strategy = 'unknown') {
        this.hideLoading();
        this.hideError();

        const contentDiv = document.getElementById('content');

        const iframe = document.createElement('iframe');
        iframe.className = 'website-frame';
        iframe.srcdoc = this.processContent(content);

        // Enable fullscreen for iframe
        iframe.setAttribute('allowfullscreen', '');
        iframe.setAttribute('webkitallowfullscreen', '');
        iframe.setAttribute('mozallowfullscreen', '');
        iframe.setAttribute('allow', 'fullscreen; autoplay; encrypted-media; picture-in-picture');

        contentDiv.innerHTML = '';

        if (strategy === 'human') {
            const notice = document.createElement('div');
            notice.className = 'mobile-notice streaming-notice';
            notice.innerHTML = '<p>🤖 Advanced loading used - site has security protection</p>';
            contentDiv.appendChild(notice);
        } else if (strategy === 'networkidle') {
            const notice = document.createElement('div');
            notice.className = 'mobile-notice';
            notice.innerHTML = '<p>⚡ Dynamic content loaded successfully</p>';
            contentDiv.appendChild(notice);
        }

        contentDiv.appendChild(iframe);

        if (title) {
            document.title = `${title} - Browser Proxy`;
        }

        this.addRefreshButton(contentDiv);
    }

    processContent(content) {
        const baseTag = `<base href="${this.currentUrl}">`;

        // Add fullscreen support script that communicates with parent
        const fullscreenScript = `
            <script>
                // Override fullscreen methods to communicate with parent
                (function() {
                    const originalRequestFullscreen = Element.prototype.requestFullscreen;
                    const originalWebkitRequestFullscreen = Element.prototype.webkitRequestFullscreen;
                    const originalMozRequestFullScreen = Element.prototype.mozRequestFullScreen;
                    
                    function requestParentFullscreen() {
                        try {
                            window.parent.postMessage({
                                type: 'requestFullscreen',
                                source: 'iframe'
                            }, '*');
                        } catch (e) {
                            console.log('Could not communicate with parent for fullscreen');
                        }
                    }
                    
                    // Override requestFullscreen
                    Element.prototype.requestFullscreen = function() {
                        requestParentFullscreen();
                        return Promise.resolve();
                    };
                    
                    if (originalWebkitRequestFullscreen) {
                        Element.prototype.webkitRequestFullscreen = function() {
                            requestParentFullscreen();
                        };
                    }
                    
                    if (originalMozRequestFullScreen) {
                        Element.prototype.mozRequestFullScreen = function() {
                            requestParentFullscreen();
                        };
                    }
                    
                    // Also handle common fullscreen button clicks
                    document.addEventListener('click', function(e) {
                        const target = e.target;
                        const isFullscreenButton = target.classList.contains('ytp-fullscreen-button') ||
                                                 target.classList.contains('fullscreen-btn') ||
                                                 target.getAttribute('data-fullscreen') ||
                                                 target.title === 'Fullscreen' ||
                                                 target.getAttribute('aria-label') === 'Fullscreen';
                        
                        if (isFullscreenButton) {
                            e.preventDefault();
                            e.stopPropagation();
                            requestParentFullscreen();
                        }
                    }, true);
                    
                    // Handle double-click on videos
                    document.addEventListener('dblclick', function(e) {
                        if (e.target.tagName === 'VIDEO') {
                            e.preventDefault();
                            requestParentFullscreen();
                        }
                    });
                })();
            </script>
        `;

        const processedContent = content.replace(
            /<head>/i,
            `<head>${baseTag}${fullscreenScript}`
        );

        return processedContent;
    }

    showLoading(message = 'Loading website...') {
        const loadingDiv = document.getElementById('loading');
        const loadingText = loadingDiv.querySelector('p');
        if (loadingText) {
            loadingText.textContent = message;
        }
        loadingDiv.classList.remove('hidden');
        this.hideError();
    }

    updateLoadingMessage(message) {
        const loadingDiv = document.getElementById('loading');
        const loadingText = loadingDiv.querySelector('p');
        if (loadingText && !loadingDiv.classList.contains('hidden')) {
            loadingText.textContent = message;
        }
    }

    hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('error').classList.remove('hidden');
        this.hideLoading();
    }

    hideError() {
        document.getElementById('error').classList.add('hidden');
    }

    addRefreshButton(contentDiv) {
        const buttonContainer = document.createElement('div');
        buttonContainer.style.position = 'fixed';
        buttonContainer.style.bottom = '20px';
        buttonContainer.style.right = '20px';
        buttonContainer.style.zIndex = '1000';
        buttonContainer.style.display = 'flex';
        buttonContainer.style.flexDirection = 'column';
        buttonContainer.style.gap = '10px';

        const refreshButton = document.createElement('button');
        refreshButton.className = 'refresh-button';
        refreshButton.innerHTML = '🔄 Refresh';
        refreshButton.onclick = () => this.loadWebsite();

        const fullscreenButton = document.createElement('button');
        fullscreenButton.className = 'fullscreen-button';
        fullscreenButton.innerHTML = '⛶ Fullscreen';
        fullscreenButton.onclick = () => this.enterFullscreenMode();

        buttonContainer.appendChild(fullscreenButton);
        buttonContainer.appendChild(refreshButton);
        contentDiv.appendChild(buttonContainer);
    }

    handleIframeMessage(event) {
        if (event.data && event.data.type === 'requestFullscreen' && event.data.source === 'iframe') {
            console.log('Received fullscreen request from iframe');
            this.enterFullscreenMode();
        }
    }

    enterFullscreenMode() {
        const iframe = document.querySelector('.website-frame');
        if (!iframe) return;

        try {
            // Try different fullscreen methods
            if (iframe.requestFullscreen) {
                iframe.requestFullscreen().catch(err => {
                    console.log('Fullscreen failed, trying document fullscreen');
                    this.fallbackFullscreen();
                });
            } else if (iframe.webkitRequestFullscreen) {
                iframe.webkitRequestFullscreen();
            } else if (iframe.mozRequestFullScreen) {
                iframe.mozRequestFullScreen();
            } else {
                this.fallbackFullscreen();
            }
        } catch (error) {
            console.log('Fullscreen request failed:', error);
            this.fallbackFullscreen();
        }
    }

    fallbackFullscreen() {
        // Fallback: make the whole document fullscreen
        const docElement = document.documentElement;
        try {
            if (docElement.requestFullscreen) {
                docElement.requestFullscreen();
            } else if (docElement.webkitRequestFullscreen) {
                docElement.webkitRequestFullscreen();
            } else if (docElement.mozRequestFullScreen) {
                docElement.mozRequestFullScreen();
            }
        } catch (error) {
            console.log('Document fullscreen also failed:', error);
        }
    }

    exitFullscreen() {
        try {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.mozCancelFullScreen) {
                document.mozCancelFullScreen();
            }
        } catch (error) {
            console.log('Exit fullscreen failed:', error);
        }
    }

    handleFullscreenChange() {
        const iframe = document.querySelector('.website-frame');
        if (iframe) {
            const isFullscreen = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement;
            if (isFullscreen) {
                console.log('Entered fullscreen mode');
                // Hide header when in fullscreen
                document.querySelector('.header').style.display = 'none';
            } else {
                console.log('Exited fullscreen mode');
                // Show header when exiting fullscreen
                document.querySelector('.header').style.display = 'block';
            }
        }
    }

    showWelcome() {
        const contentDiv = document.getElementById('content');
        contentDiv.innerHTML = `
            <div class="welcome">
                <h1>Universal Web Browser</h1>
                <p>Browse any website through your laptop - from simple sites like Google to complex streaming platforms.</p>
                <p>✅ Basic websites (Google, Wikipedia, news sites)</p>
                <p>✅ Complex sites with security layers</p>
                <p>✅ Streaming and video content</p>
                <p>✅ Mobile-optimized versions</p>
                <p>First, configure your laptop's IP address in settings.</p>
            </div>
        `;
        this.hideLoading();
        this.hideError();
        document.title = 'Universal Web Browser';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BrowserProxyApp();
});

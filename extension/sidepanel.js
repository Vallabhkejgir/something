document.addEventListener('DOMContentLoaded', () => {
  const indexBtn = document.getElementById('indexBtn');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const chatContainer = document.getElementById('chatContainer');
  const loadingBubble = document.getElementById('loadingBubble');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  let currentTabUrl = '';

  // Get active tab URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    if (tabs.length === 0) return;
    const tab = tabs[0];
    currentTabUrl = tab.url;
    
    // Check if backend already indexed this URL
    checkStatus(currentTabUrl);
  });

  async function checkStatus(url) {
    try {
      const res = await fetch(`http://localhost:5000/api/status?url=${encodeURIComponent(url)}`);
      const data = await res.json();
      if (data.initialized) {
        setReady();
      } else {
        setUninitialized();
      }
    } catch (error) {
      setUninitialized();
      appendMessage('system', 'Cannot connect to backend server at http://localhost:5000');
    }
  }

  function setReady() {
    statusDot.className = 'dot ready';
    statusText.textContent = 'Page Indexed';
    indexBtn.disabled = true;
    indexBtn.textContent = 'Indexed';
    chatInput.disabled = false;
    sendBtn.disabled = false;
  }

  function setUninitialized() {
    statusDot.className = 'dot';
    statusText.textContent = 'Not Indexed';
    indexBtn.disabled = false;
    indexBtn.textContent = 'Index Current Page';
    chatInput.disabled = true;
    sendBtn.disabled = true;
  }

  function setIndexing() {
    statusDot.className = 'dot indexing';
    statusText.textContent = 'Indexing page...';
    indexBtn.disabled = true;
    indexBtn.textContent = 'Extracting and Indexing...';
  }

  indexBtn.addEventListener('click', async () => {
    setIndexing();
    
    // Inject content script to extract page content
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs[0];
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content.js']
      }, () => {
        if (chrome.runtime.lastError) {
          setUninitialized();
          appendMessage('system', 'Cannot index this page (e.g. chrome:// pages are not allowed): ' + chrome.runtime.lastError.message);
          return;
        }
        // After injection, wait a bit then ask for content
        setTimeout(() => {
          chrome.tabs.sendMessage(tab.id, { action: "extract" }, async (response) => {
            if (chrome.runtime.lastError) {
              setUninitialized();
              appendMessage('system', 'Error extracting page content: ' + chrome.runtime.lastError.message);
              return;
            }
            if (!response || !response.elements) {
              setUninitialized();
              appendMessage('system', 'Failed to extract content.');
              return;
            }

            try {
              const res = await fetch('http://localhost:5000/api/initialize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  url: currentTabUrl,
                  title: tab.title,
                  elements: response.elements
                })
              });
              
              if (res.ok) {
                setReady();
                appendMessage('system', 'Successfully indexed text, tables, and images. You can now ask questions!');
              } else {
                const err = await res.json();
                setUninitialized();
                appendMessage('system', 'Failed to index: ' + err.error);
              }
            } catch (err) {
              setUninitialized();
              appendMessage('system', 'Failed to send to backend: ' + err.message);
            }
          });
        }, 500); // small delay to ensure content script is ready
      });
    });
  });

  sendBtn.addEventListener('click', sendQuery);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuery();
  });

  async function sendQuery() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;
    
    // Show typing bubble
    chatContainer.appendChild(loadingBubble);
    loadingBubble.style.display = 'block';
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
      const res = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text })
      });
      
      loadingBubble.style.display = 'none';
      
      if (res.ok) {
        const data = await res.json();
        appendMessage('assistant', data.answer, true);
      } else {
        const err = await res.json();
        appendMessage('system', 'Error: ' + err.error);
      }
    } catch (err) {
      loadingBubble.style.display = 'none';
      appendMessage('system', 'Error connecting to backend.');
    } finally {
      chatInput.disabled = false;
      sendBtn.disabled = false;
      chatInput.focus();
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  function appendMessage(role, text, parseMarkdown = false) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    if (parseMarkdown && window.marked) {
      div.innerHTML = marked.parse(text);
    } else {
      div.textContent = text;
    }
    
    if (role !== 'system') {
      // Put message before the loading bubble
      if (loadingBubble.parentNode === chatContainer) {
        chatContainer.insertBefore(div, loadingBubble);
      } else {
        chatContainer.appendChild(div);
      }
    } else {
      div.className = 'message assistant';
      div.style.backgroundColor = '#fee2e2';
      div.style.borderColor = '#f87171';
      div.style.color = '#991b1b';
      chatContainer.appendChild(div);
    }
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
});

// Only add listener if it hasn't been added yet
if (!window.contentScriptLoaded) {
  window.contentScriptLoaded = true;

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extract") {
      try {
        const elements = extractPageContent();
        sendResponse({ elements });
      } catch (err) {
        console.error("Extraction error:", err);
        sendResponse({ elements: [], error: err.toString() });
      }
    }
    return true; // Keep channel open for async response if needed
  });

  function extractPageContent() {
    const elements = [];

    // 1. Extract Main Text
    // We try to grab the main readable content instead of scripts/styles
    const bodyClone = document.body.cloneNode(true);
    // Remove unwanted elements
    const unwanted = bodyClone.querySelectorAll('script, style, nav, footer, iframe, noscript, svg, button');
    unwanted.forEach(el => el.remove());
    
    // Extract text in chunks (e.g. by paragraphs or headings)
    const textNodes = bodyClone.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li');
    let currentText = [];
    textNodes.forEach(node => {
      const text = node.textContent.trim();
      if (text.length > 20) {
        currentText.push(text);
      }
    });
    
    if (currentText.length > 0) {
      elements.push({
        type: 'text',
        content: currentText.join('\n\n')
      });
    }

    // 2. Extract Tables
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
      const mdTable = convertTableToMarkdown(table);
      if (mdTable) {
        elements.push({
          type: 'table',
          content: mdTable
        });
      }
    });

    // 3. Extract Images
    const images = document.querySelectorAll('img');
    images.forEach(img => {
      // Ignore tiny images (icons)
      if (img.width < 100 || img.height < 100) return;
      
      const src = img.src;
      const alt = img.alt || img.title || "No description provided";
      
      // We could draw it to canvas to get base64 if needed, 
      // but passing URL might be enough if backend can fetch.
      // Since some images are behind auth, base64 is safer.
      const base64 = getBase64Image(img);
      
      if (src && base64) {
        elements.push({
          type: 'image',
          url: src,
          alt: alt,
          base64: base64
        });
      }
    });

    return elements;
  }

  function convertTableToMarkdown(table) {
    let md = "";
    const rows = table.querySelectorAll('tr');
    if (rows.length === 0) return null;

    rows.forEach((row, i) => {
      const cols = row.querySelectorAll('th, td');
      let rowMd = "|";
      cols.forEach(col => {
        let text = col.textContent.replace(/\n/g, ' ').replace(/\|/g, '\\|').trim();
        rowMd += ` ${text} |`;
      });
      md += rowMd + "\n";
      
      // Add separator after headers
      if (i === 0) {
        let sepMd = "|";
        cols.forEach(() => {
          sepMd += "---|";
        });
        md += sepMd + "\n";
      }
    });
    return md;
  }

  function getBase64Image(img) {
    try {
      // If image is not fully loaded or cross-origin, this might fail or return blank
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      return canvas.toDataURL("image/png");
    } catch (e) {
      return null;
    }
  }
}

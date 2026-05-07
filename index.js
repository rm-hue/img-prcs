const input = document.getElementById('imgInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const results = document.getElementById('results');
let currentFile = null;

// Handle file selection and preview
input.onchange = () => {
    const file = input.files[0];
    if (file) {
        currentFile = file;
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block'; 
        predictBtn.disabled = false;
        results.innerHTML = 'Image loaded. Click "Predict" to analyze.';
    }
};

// Handle the prediction request
predictBtn.onclick = async () => {
    if (!currentFile) return;

    // UI Feedback
    results.innerHTML = '<span style="color: blue;">Processing image...</span>';
    predictBtn.disabled = true;

    try {
        const fd = new FormData();
        fd.append('image', currentFile);

        // Use relative URL if HTML is served by Flask, or keep 127.0.0.1 if opening HTML directly
        const response = await fetch('http://127.0.0.1:5000/predict', { 
            method: 'POST', 
            body: fd 
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            results.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
        } else if (data.prediction) {
            const p = data.prediction;
            const confidencePercent = (p.confidence * 100).toFixed(2);
            
            // Choose color based on status
            const statusColor = p.status === 'BAD' ? 'red' : 'green';
            
            // FIXED: Using the correct variable names that app.py actually sends
            results.innerHTML = `
                <div style="margin-top: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                    <div style="margin-bottom: 8px;">
                        <strong>Product Category:</strong> ${p.product.charAt(0).toUpperCase() + p.product.slice(1)}
                    </div>
                    <div style="margin-bottom: 8px;">
                        <strong>Quality Status:</strong> 
                        <span style="color: ${statusColor}; font-weight: bold; font-size: 1.1em;">${p.status}</span>
                    </div>
                    <div>
                        <strong>Confidence:</strong> ${confidencePercent}%
                    </div>
                </div>
            `;
        } else {
            results.innerHTML = 'No prediction data received from server.';
        }

    } catch (error) {
        console.error('Fetch error:', error);
        results.innerHTML = `<span style="color: red;">Error: Could not connect to backend. Make sure app.py is running.</span>`;
    } finally {
        predictBtn.disabled = false;
    }
};
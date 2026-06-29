function showToast(message, type = 'success') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;

    const toastId = `toast-${Date.now()}`;
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center border-0 glass-toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas ${type === 'success' ? 'fa-check-circle text-success' : 'fa-exclamation-circle text-danger'} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    const toastEl = document.getElementById(toastId);
    const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });
    bsToast.show();

    toastEl.addEventListener('hidden.bs.toast', () => {
        toastEl.remove();
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        const currentTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', currentTheme);
        themeToggle.checked = currentTheme === 'light';

        themeToggle.addEventListener('change', (e) => {
            const newTheme = e.target.checked ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            showToast(`Switched to ${newTheme} mode`, 'success');
        });
    }

    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const uploadPrompt = document.getElementById('upload-prompt');
    const imagePreview = document.getElementById('image-preview');
    const btnRemove = document.getElementById('btn-remove-image');
    const btnPredict = document.getElementById('btn-predict');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsPanel = document.getElementById('results-panel');

    let selectedFile = null;

    if (uploadZone && fileInput) {
        uploadZone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
            }, false);
        });

        uploadZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files.length > 0) {
                handleFileSelect(dt.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            showToast('Please select an image file (JPG, PNG)', 'danger');
            return;
        }
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('d-none');
            previewContainer.classList.remove('d-none');
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (btnRemove) {
        btnRemove.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedFile = null;
            fileInput.value = '';
            imagePreview.src = '';
            previewContainer.classList.add('d-none');
            uploadPrompt.classList.remove('d-none');
            btnPredict.disabled = true;
            resultsPanel.classList.add('d-none');
        });
    }

    if (btnPredict) {
        btnPredict.addEventListener('click', async () => {
            if (!selectedFile) return;

            loadingOverlay.classList.remove('d-none');
            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) throw new Error('Prediction request failed');

                const data = await res.json();
                renderDetectionResults(data);
                showToast('Object detection completed successfully!', 'success');
            } catch (err) {
                console.error(err);
                showToast('Error running object detection model', 'danger');
            } finally {
                loadingOverlay.classList.add('d-none');
            }
        });
    }

    function renderDetectionResults(data) {
        resultsPanel.classList.remove('d-none');
        resultsPanel.scrollIntoView({ behavior: 'smooth' });

        document.getElementById('res-annotated-image').src = data.output_filename;
        document.getElementById('res-total-objects').textContent = data.objects_count;
        document.getElementById('res-model-name').textContent = data.model_name;
        document.getElementById('res-inference-time').textContent = `${data.inference_time} ms`;
        document.getElementById('res-avg-confidence').textContent = `${data.average_confidence}%`;

        const listContainer = document.getElementById('res-objects-list');
        listContainer.innerHTML = '';

        if (data.objects_list.length === 0) {
            listContainer.innerHTML = '<li class="list-group-item bg-transparent text-muted">No objects detected.</li>';
        } else {
            const counts = {};
            data.objects_list.forEach(obj => {
                counts[obj.name] = (counts[obj.name] || 0) + 1;
            });

            data.objects_list.forEach(obj => {
                const confPercent = Math.round(obj.confidence * 100);
                const li = document.createElement('li');
                li.className = 'list-group-item bg-transparent d-flex justify-content-between align-items-center border-0 border-bottom border-secondary-subtle px-0 py-2';
                li.innerHTML = `
                    <span>
                        <i class="fas fa-tag text-cyan me-2"></i>
                        <strong>${obj.name.toUpperCase()}</strong>
                    </span>
                    <span class="badge bg-opacity-10 bg-cyan text-cyan rounded-pill px-3 py-1.5" style="background-color: rgba(0,242,254,0.15)">
                        ${confPercent}% Conf.
                    </span>
                `;
                listContainer.appendChild(li);
            });
        }

        const btnDownloadImage = document.getElementById('btn-download-image');
        if (btnDownloadImage) {
            btnDownloadImage.onclick = () => {
                const a = document.createElement('a');
                a.href = data.output_filename;
                a.download = `detected_${Date.now()}.jpg`;
                a.click();
            };
        }

        const btnCopyJson = document.getElementById('btn-copy-json');
        if (btnCopyJson) {
            btnCopyJson.onclick = () => {
                navigator.clipboard.writeText(JSON.stringify(data, null, 2))
                    .then(() => showToast('JSON results copied to clipboard', 'success'))
                    .catch(() => showToast('Failed to copy text', 'danger'));
            };
        }

        const btnPrintReport = document.getElementById('btn-print-report');
        if (btnPrintReport) {
            btnPrintReport.onclick = () => {
                const printWindow = window.open('', '_blank');
                const html = `
                    <html>
                    <head>
                        <title>AI Object Detection Report</title>
                        <style>
                            body { font-family: Arial, sans-serif; padding: 40px; color: #333; }
                            h1 { color: #0088cc; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
                            .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }
                            .meta-item { background: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #eee; }
                            .img-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }
                            img { width: 100%; border-radius: 8px; border: 1px solid #ccc; }
                            ul { padding-left: 20px; }
                            li { margin-bottom: 8px; }
                        </style>
                    </head>
                    <body>
                        <h1>AI Object Detection Report</h1>
                        <p>Date: ${new Date().toLocaleString()}</p>
                        <div class="img-container">
                            <div>
                                <h3>Original Image</h3>
                                <img src="${window.location.origin}${data.filename}" />
                            </div>
                            <div>
                                <h3>Detection Output</h3>
                                <img src="${window.location.origin}${data.output_filename}" />
                            </div>
                        </div>
                        <div class="meta-grid">
                            <div class="meta-item">
                                <strong>Model Name:</strong> ${data.model_name}<br>
                                <strong>Inference Time:</strong> ${data.inference_time} ms<br>
                                <strong>Average Confidence:</strong> ${data.average_confidence}%
                            </div>
                            <div class="meta-item">
                                <strong>Total Objects Detected:</strong> ${data.objects_count}
                            </div>
                        </div>
                        <h3>Detections Log:</h3>
                        <ul>
                            ${data.objects_list.map(obj => `<li><strong>${obj.name.toUpperCase()}</strong>: Confidence ${Math.round(obj.confidence * 100)}% (Box: ${obj.box.join(', ')})</li>`).join('')}
                        </ul>
                        <script>window.onload = function() { window.print(); }</script>
                    </body>
                    </html>
                `;
                printWindow.document.write(html);
                printWindow.document.close();
            };
        }
    }

});


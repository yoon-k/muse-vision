/**
 * MUSE Vision - Interactive Demo
 * Object Detection, Face Recognition, Image Search
 */

const VisionDemo = {
    currentTab: 'detection',

    // Sample detection results (simulated)
    sampleDetections: {
        street: [
            { class: 'person', confidence: 0.95, color: '#22c55e' },
            { class: 'person', confidence: 0.91, color: '#22c55e' },
            { class: 'car', confidence: 0.89, color: '#3b82f6' },
            { class: 'car', confidence: 0.87, color: '#3b82f6' },
            { class: 'bicycle', confidence: 0.82, color: '#f59e0b' },
            { class: 'traffic light', confidence: 0.94, color: '#ef4444' }
        ],
        office: [
            { class: 'person', confidence: 0.97, color: '#22c55e' },
            { class: 'laptop', confidence: 0.93, color: '#3b82f6' },
            { class: 'chair', confidence: 0.88, color: '#f59e0b' },
            { class: 'desk', confidence: 0.85, color: '#f59e0b' },
            { class: 'monitor', confidence: 0.91, color: '#3b82f6' }
        ],
        pets: [
            { class: 'dog', confidence: 0.96, color: '#f59e0b' },
            { class: 'cat', confidence: 0.93, color: '#f59e0b' },
            { class: 'person', confidence: 0.89, color: '#22c55e' },
            { class: 'couch', confidence: 0.84, color: '#3b82f6' }
        ]
    },

    sampleFaces: {
        group: [
            { id: 1, age: '25-30', gender: 'Male', emotion: 'Happy', confidence: 0.94 },
            { id: 2, age: '30-35', gender: 'Female', emotion: 'Neutral', confidence: 0.92 },
            { id: 3, age: '20-25', gender: 'Male', emotion: 'Happy', confidence: 0.89 },
            { id: 4, age: '35-40', gender: 'Female', emotion: 'Happy', confidence: 0.91 }
        ],
        portrait: [
            { id: 1, age: '28-32', gender: 'Female', emotion: 'Neutral', confidence: 0.97 }
        ]
    },

    searchResults: [
        { similarity: 0.94, category: 'similar' },
        { similarity: 0.91, category: 'similar' },
        { similarity: 0.87, category: 'related' },
        { similarity: 0.84, category: 'related' },
        { similarity: 0.81, category: 'related' },
        { similarity: 0.78, category: 'related' }
    ],

    init() {
        this.setupTabs();
        this.setupUploads();
        this.setupSearch();
    },

    setupTabs() {
        document.querySelectorAll('.demo-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });
    },

    switchTab(tabName) {
        this.currentTab = tabName;

        document.querySelectorAll('.demo-tab').forEach(tab => {
            tab.classList.toggle('active', tab.getAttribute('data-tab') === tabName);
        });

        document.querySelectorAll('.demo-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${tabName}-panel`);
        });
    },

    setupUploads() {
        // Detection upload
        const detectionUpload = document.getElementById('detection-upload');
        const detectionInput = document.getElementById('detection-input');

        if (detectionUpload && detectionInput) {
            detectionUpload.addEventListener('click', () => detectionInput.click());
            detectionInput.addEventListener('change', (e) => this.handleImageUpload(e, 'detection'));
            this.setupDragDrop(detectionUpload, 'detection');
        }

        // Face upload
        const faceUpload = document.getElementById('face-upload');
        const faceInput = document.getElementById('face-input');

        if (faceUpload && faceInput) {
            faceUpload.addEventListener('click', () => faceInput.click());
            faceInput.addEventListener('change', (e) => this.handleImageUpload(e, 'face'));
            this.setupDragDrop(faceUpload, 'face');
        }
    },

    setupDragDrop(element, type) {
        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            element.style.background = 'rgba(0, 212, 255, 0.1)';
        });

        element.addEventListener('dragleave', () => {
            element.style.background = '';
        });

        element.addEventListener('drop', (e) => {
            e.preventDefault();
            element.style.background = '';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.processImage(file, type);
            }
        });
    },

    handleImageUpload(event, type) {
        const file = event.target.files[0];
        if (file) {
            this.processImage(file, type);
        }
    },

    processImage(file, type) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (type === 'detection') {
                this.showDetectionResults(e.target.result);
            } else if (type === 'face') {
                this.showFaceResults(e.target.result);
            }
        };
        reader.readAsDataURL(file);
    },

    showDetectionResults(imageSrc) {
        const uploadArea = document.getElementById('detection-upload');
        const preview = document.getElementById('detection-preview');
        const canvas = document.getElementById('detection-canvas');
        const resultsList = document.getElementById('detection-list');
        const timeDisplay = document.getElementById('detection-time');

        // Show loading
        resultsList.innerHTML = '<div class="loading">Analyzing image...</div>';

        // Display image
        preview.src = imageSrc;
        preview.style.display = 'block';
        preview.onload = () => {
            const placeholder = uploadArea.querySelector('.upload-placeholder');
            if (placeholder) placeholder.style.display = 'none';

            // Simulate processing delay
            setTimeout(() => {
                const detections = this.generateRandomDetections();
                timeDisplay.textContent = `${Math.floor(Math.random() * 20 + 10)}ms`;
                this.renderDetections(detections, resultsList);
                this.drawBoundingBoxes(preview, canvas, detections);
            }, 800);
        };
    },

    generateRandomDetections() {
        const allObjects = [
            { class: 'person', color: '#22c55e' },
            { class: 'car', color: '#3b82f6' },
            { class: 'dog', color: '#f59e0b' },
            { class: 'cat', color: '#f59e0b' },
            { class: 'chair', color: '#3b82f6' },
            { class: 'laptop', color: '#3b82f6' },
            { class: 'phone', color: '#3b82f6' },
            { class: 'book', color: '#8b5cf6' }
        ];

        const count = Math.floor(Math.random() * 4) + 2;
        const selected = [];
        for (let i = 0; i < count; i++) {
            const obj = allObjects[Math.floor(Math.random() * allObjects.length)];
            selected.push({
                ...obj,
                confidence: 0.75 + Math.random() * 0.2
            });
        }
        return selected.sort((a, b) => b.confidence - a.confidence);
    },

    renderDetections(detections, container) {
        container.innerHTML = detections.map(d => `
            <div class="detection-item">
                <span class="detection-badge" style="background: ${d.color}20; color: ${d.color}">${d.class}</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${d.confidence * 100}%; background: ${d.color}"></div>
                </div>
                <span class="confidence-value">${(d.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    },

    drawBoundingBoxes(img, canvas, detections) {
        canvas.style.display = 'block';
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.width = '100%';
        canvas.style.height = '100%';

        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        detections.forEach((d, i) => {
            const x = Math.random() * (canvas.width * 0.5) + canvas.width * 0.1;
            const y = Math.random() * (canvas.height * 0.5) + canvas.height * 0.1;
            const w = Math.random() * (canvas.width * 0.3) + canvas.width * 0.1;
            const h = Math.random() * (canvas.height * 0.3) + canvas.height * 0.1;

            ctx.strokeStyle = d.color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = d.color;
            ctx.fillRect(x, y - 25, ctx.measureText(d.class).width + 20, 25);

            ctx.fillStyle = '#fff';
            ctx.font = 'bold 14px Inter';
            ctx.fillText(d.class, x + 5, y - 7);
        });
    },

    showFaceResults(imageSrc) {
        const uploadArea = document.getElementById('face-upload');
        const preview = document.getElementById('face-preview');
        const canvas = document.getElementById('face-canvas');
        const resultsList = document.getElementById('face-list');
        const timeDisplay = document.getElementById('face-time');

        resultsList.innerHTML = '<div class="loading">Analyzing faces...</div>';

        preview.src = imageSrc;
        preview.style.display = 'block';
        preview.onload = () => {
            const placeholder = uploadArea.querySelector('.upload-placeholder');
            if (placeholder) placeholder.style.display = 'none';

            setTimeout(() => {
                const faces = this.generateRandomFaces();
                timeDisplay.textContent = `${Math.floor(Math.random() * 50 + 80)}ms`;
                this.renderFaces(faces, resultsList);
                this.drawFaceBoxes(preview, canvas, faces);
            }, 1000);
        };
    },

    generateRandomFaces() {
        const count = Math.floor(Math.random() * 3) + 1;
        const faces = [];
        const ages = ['18-25', '25-30', '30-35', '35-40', '40-50'];
        const emotions = ['Happy', 'Neutral', 'Surprised', 'Calm'];

        for (let i = 0; i < count; i++) {
            faces.push({
                id: i + 1,
                age: ages[Math.floor(Math.random() * ages.length)],
                gender: Math.random() > 0.5 ? 'Male' : 'Female',
                emotion: emotions[Math.floor(Math.random() * emotions.length)],
                confidence: 0.85 + Math.random() * 0.12
            });
        }
        return faces;
    },

    renderFaces(faces, container) {
        container.innerHTML = faces.map(f => `
            <div class="face-item">
                <div class="face-info">
                    <div class="face-id">Face #${f.id}</div>
                    <div class="face-details">
                        <span>${f.age}</span> ·
                        <span>${f.gender}</span> ·
                        <span>${f.emotion}</span>
                    </div>
                </div>
                <span class="confidence-value">${(f.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    },

    drawFaceBoxes(img, canvas, faces) {
        canvas.style.display = 'block';
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.width = '100%';
        canvas.style.height = '100%';

        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        faces.forEach((f, i) => {
            const x = (canvas.width / (faces.length + 1)) * (i + 1) - 50;
            const y = canvas.height * 0.2;
            const size = Math.min(canvas.width, canvas.height) * 0.25;

            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, size, size * 1.2);

            ctx.fillStyle = '#f59e0b';
            ctx.fillRect(x, y + size * 1.2, size, 25);

            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px Inter';
            ctx.fillText(`Face #${f.id} - ${f.emotion}`, x + 5, y + size * 1.2 + 17);
        });
    },

    setupSearch() {
        const textMode = document.querySelector('.search-mode[data-mode="text"]');
        const imageMode = document.querySelector('.search-mode[data-mode="image"]');
        const textArea = document.getElementById('text-search-area');
        const imageArea = document.getElementById('image-search-area');
        const searchInput = document.getElementById('search-query');

        if (textMode && imageMode) {
            textMode.addEventListener('click', () => {
                textMode.classList.add('active');
                imageMode.classList.remove('active');
                textArea.classList.add('active');
                imageArea.classList.remove('active');
            });

            imageMode.addEventListener('click', () => {
                imageMode.classList.add('active');
                textMode.classList.remove('active');
                imageArea.classList.add('active');
                textArea.classList.remove('active');
            });
        }

        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
        }

        const imageUploadMini = document.querySelector('.image-upload-mini');
        const searchImageInput = document.getElementById('search-image-input');

        if (imageUploadMini && searchImageInput) {
            imageUploadMini.addEventListener('click', () => searchImageInput.click());
            searchImageInput.addEventListener('change', () => {
                performImageSearch();
            });
        }
    }
};

// Global functions
function loadSampleImage(type, sample) {
    const resultsList = document.getElementById(`${type}-list`);
    const timeDisplay = document.getElementById(`${type}-time`);

    if (type === 'detection') {
        resultsList.innerHTML = '<div class="loading">Loading sample...</div>';
        setTimeout(() => {
            const detections = VisionDemo.sampleDetections[sample] || VisionDemo.generateRandomDetections();
            timeDisplay.textContent = `${Math.floor(Math.random() * 15 + 8)}ms`;
            VisionDemo.renderDetections(detections, resultsList);
        }, 600);
    } else if (type === 'face') {
        resultsList.innerHTML = '<div class="loading">Loading sample...</div>';
        setTimeout(() => {
            const faces = VisionDemo.sampleFaces[sample] || VisionDemo.generateRandomFaces();
            timeDisplay.textContent = `${Math.floor(Math.random() * 40 + 60)}ms`;
            VisionDemo.renderFaces(faces, resultsList);
        }, 800);
    }
}

function performSearch() {
    const query = document.getElementById('search-query').value.trim();
    if (!query) return;

    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '<div class="loading" style="grid-column: 1/-1; text-align: center; padding: 40px;">Searching...</div>';

    setTimeout(() => {
        showSearchResults(query);
    }, 1200);
}

function searchSuggestion(query) {
    document.getElementById('search-query').value = query;
    performSearch();
}

function performImageSearch() {
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '<div class="loading" style="grid-column: 1/-1; text-align: center; padding: 40px;">Finding similar images...</div>';

    setTimeout(() => {
        showSearchResults('image query');
    }, 1500);
}

function showSearchResults(query) {
    const resultsContainer = document.getElementById('search-results');
    const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

    resultsContainer.innerHTML = VisionDemo.searchResults.map((r, i) => `
        <div class="search-result-item" style="background: linear-gradient(135deg, ${colors[i % colors.length]}40, ${colors[(i+1) % colors.length]}20);">
            <div style="position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 48px; opacity: 0.3;">
                ${['🏔️', '🐕', '🌅', '🏙️', '🌊', '🌸'][i % 6]}
            </div>
            <span class="similarity-badge">${(r.similarity * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    VisionDemo.init();
});

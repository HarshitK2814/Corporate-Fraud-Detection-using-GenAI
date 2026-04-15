
let fetchedImages = [];
let fetchedMetadata = {};
let currentVoiceInput = 'upload';
let map, marker; // Leaflet vars

document.addEventListener('DOMContentLoaded', () => {
    restoreState();
    initMap(); // Initialize Map

    // Add Sidebar Navigation
    document.querySelectorAll('.sidebar li').forEach(li => {
        li.addEventListener('click', () => {
            const step = parseInt(li.dataset.step);
            if (!isNaN(step)) goToStep(step);
        });
    });
});

function initMap() {
    const latInput = document.getElementById('lat');
    const lonInput = document.getElementById('lon');

    // Default to Infosys logic if empty
    const lat = latInput ? parseFloat(latInput.value) : 12.8399;
    const lon = lonInput ? parseFloat(lonInput.value) : 77.6770;

    const mapContainer = document.getElementById('map-container');
    if (mapContainer && !map) {
        map = L.map('map-container').setView([lat, lon], 16);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        marker = L.marker([lat, lon]).addTo(map);
    }
}

function goToStep(step) {
    // 1. Update Sidebar
    document.querySelectorAll('.sidebar li').forEach(li => {
        li.classList.remove('active');
        if (parseInt(li.dataset.step) === step) li.classList.add('active');
    });

    // 2. Show/Hide Steps (Inline Style Verification Fix)
    document.querySelectorAll('.step-card').forEach(card => {
        card.classList.add('hidden');
        card.classList.remove('active');
        card.style.display = 'none'; // Force hide
    });

    const target = document.getElementById(`step-${step}`);
    if (target) {
        target.classList.remove('hidden');
        target.style.display = 'block'; // Force show

        // Slight delay to allow display:block to kick in for transition
        setTimeout(() => {
            target.classList.add('active');
            // Resize map if entering step 2
            if (step === 2 && map) {
                map.invalidateSize();
            }
        }, 10);
    }

    // 3. Save State
    localStorage.setItem('step', step);
}

function saveState() {
    const company = document.getElementById('company-name');
    const lat = document.getElementById('lat');
    const lon = document.getElementById('lon');

    if (company) localStorage.setItem('company', company.value);
    if (lat) localStorage.setItem('lat', lat.value);
    if (lon) localStorage.setItem('lon', lon.value);

    // Save images/metadata
    if (fetchedImages.length > 0) localStorage.setItem('images', JSON.stringify(fetchedImages));
    if (fetchedMetadata && Object.keys(fetchedMetadata).length > 0) localStorage.setItem('metadata', JSON.stringify(fetchedMetadata));
}

function restoreState() {
    if (localStorage.getItem('company')) {
        const el = document.getElementById('company-name');
        if (el) el.value = localStorage.getItem('company');
    }
    if (localStorage.getItem('lat')) {
        const el = document.getElementById('lat');
        if (el) el.value = localStorage.getItem('lat');
    }
    if (localStorage.getItem('lon')) {
        const el = document.getElementById('lon');
        if (el) el.value = localStorage.getItem('lon');
    }

    const savedImages = localStorage.getItem('images');
    if (savedImages) {
        try { fetchedImages = JSON.parse(savedImages); } catch (e) { }
    }
    const savedMeta = localStorage.getItem('metadata');
    if (savedMeta) {
        try { fetchedMetadata = JSON.parse(savedMeta); } catch (e) { }
    }

    // Restore UI for Step 2 if data exists
    if (fetchedImages.length > 0) {
        renderImages(fetchedImages);
        const btnNext3 = document.getElementById('btn-next-3');
        if (btnNext3) btnNext3.classList.remove('hidden');
    }

    const savedStep = localStorage.getItem('step');
    if (savedStep) goToStep(parseInt(savedStep));

    // Attach listeners
    ['company-name', 'lat', 'lon'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', saveState);
    });
}

// ── Step 2: Geospatial ─────────────────────────────────────────────────────

function runGeospatialFetch() {
    const company = document.getElementById('company-name').value;
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;

    if (!company || !lat || !lon) { alert("Please fill in all fields"); return; }

    // Update Map
    if (map && marker) {
        const newLat = parseFloat(lat);
        const newLon = parseFloat(lon);
        map.flyTo([newLat, newLon], 16);
        marker.setLatLng([newLat, newLon]);
    }

    const btn = document.querySelector('#step-2 .btn-secondary');
    if (btn) {
        btn.innerText = "Fetch & Verifying...";
        btn.disabled = true;
    }

    fetch('/api/verify_geo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company, lat, lon })
    })
        .then(r => r.json())
        .then(data => {
            if (btn) {
                btn.innerText = "📍 Verify Coordinates";
                btn.disabled = false;
            }

            if (data.image_urls) {
                fetchedImages = data.image_urls;
                fetchedMetadata = data.metadata || {};
                saveState();
                renderImages(fetchedImages);
                const nextBtn = document.getElementById('btn-next-3');
                if (nextBtn) nextBtn.classList.remove('hidden');
            }
        })
        .catch(e => {
            console.error(e);
            if (btn) {
                btn.innerText = "Error (Try Again)";
                btn.disabled = false;
            }
        });
}

function renderImages(urls) {
    const grid = document.getElementById('geo-results');
    if (!grid) return;
    grid.innerHTML = '';
    grid.classList.remove('hidden');

    // Metadata / Status Box
    if (fetchedMetadata) {
        const score = fetchedMetadata.match_score || 0;
        const foundName = fetchedMetadata.found_place_name || "Unknown";
        const isVerified = fetchedMetadata.verified_entity;

        const statusBox = document.createElement('div');
        statusBox.style.padding = '10px';
        statusBox.style.borderRadius = '8px';
        statusBox.style.marginBottom = '15px';
        statusBox.style.textAlign = 'center';
        statusBox.style.fontSize = '0.9rem';
        statusBox.style.gridColumn = "1 / -1"; // Full width

        if (isVerified) {
            if (score > 0.8) {
                statusBox.style.background = '#dcfce7';
                statusBox.style.color = '#166534';
                statusBox.innerHTML = `✅ <strong>Verified Entity:</strong> "${foundName}" (Exact Match: ${(score * 100).toFixed(0)}%)`;
            } else {
                statusBox.style.background = '#ffedd5';
                statusBox.style.color = '#9a3412';
                statusBox.innerHTML = `⚠️ <strong>Partial Match:</strong> Searched for "${document.getElementById('company-name').value}", found "<strong>${foundName}</strong>" (Match: ${(score * 100).toFixed(0)}%)`;
            }
        } else {
            statusBox.style.background = '#fee2e2';
            statusBox.style.color = '#991b1b';
            statusBox.innerHTML = `🚨 <strong>Name Mismatch:</strong> Found "${foundName}" vs Input (Match: ${(score * 100).toFixed(0)}%). <strong>High Risk.</strong>`;
        }
        grid.appendChild(statusBox);
    }

    urls.forEach(url => {
        const div = document.createElement('div');
        div.className = 'img-card';
        div.innerHTML = `<img src="${url}" onclick="window.open('${url}')" alt="Building">`;
        grid.appendChild(div);
    });
}

// ── Step 3: Classification ─────────────────────────────────────────────────

function runClassification() {
    const loading = document.getElementById('classify-loading');
    const results = document.getElementById('classification-results');

    if (loading) loading.classList.remove('hidden');
    if (results) results.classList.add('hidden');

    fetch('/api/classify_building', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_urls: fetchedImages,
            verified_entity: fetchedMetadata.verified_entity || false
        })
    })
        .then(r => r.json())
        .then(data => {
            if (loading) loading.classList.add('hidden');
            if (results) results.classList.remove('hidden');

            // Fix API mismatch & Save for Fusion
            const finalRisk = data.avg_shell_risk !== undefined ? data.avg_shell_risk : 0.5;
            const reasoning = data.verdict || "Analysis Complete";

            fetchedMetadata.geo_risk = finalRisk;
            fetchedMetadata.geo_verdict = reasoning;
            saveState();

            let color = '#fbbf24'; // yellow
            if (finalRisk > 0.7) color = '#f87171'; // red
            if (finalRisk < 0.3) color = '#4ade80'; // green

            results.innerHTML = `
            <h3>Risk Assessment</h3>
            <div class="verdict" style="background:${color}22; color:${color}; border:1px solid ${color}">
                Risk Score: ${(finalRisk * 100).toFixed(0)}%
            </div>
            <p style="margin-top:10px">${reasoning}</p>
        `;

            const nextBtn = document.getElementById('btn-next-4');
            if (nextBtn) nextBtn.classList.remove('hidden');
        })
        .catch(e => {
            if (loading) loading.classList.add('hidden');
            alert("Classification failed: " + e);
        });
}

// ── Step 4: Voice Analysis ────────────────────────────────────────────────

function switchTab(tab) {
    currentVoiceInput = tab;
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    const target = document.getElementById(`tab-${tab}`);
    if (target) target.classList.remove('hidden');

    const btns = document.querySelectorAll('.tab-btn');
    if (tab === 'upload' && btns[0]) btns[0].classList.add('active');
    if (tab === 'youtube' && btns[1]) btns[1].classList.add('active');
    if (tab === 'text' && btns[2]) btns[2].classList.add('active');
}

function runVoiceAnalysis() {
    const btn = document.querySelector('#step-4 .btn-secondary');
    const loading = document.getElementById('voice-loading');
    const results = document.getElementById('voice-results');

    if (btn) btn.classList.add('hidden');
    if (loading) loading.classList.remove('hidden');
    if (results) results.classList.add('hidden');

    const formData = new FormData();
    formData.append('input_type', currentVoiceInput);

    if (currentVoiceInput === 'upload') {
        const fileInput = document.getElementById('audio-file');
        if (fileInput.files.length === 0) { alert("Please select a file first."); resetVoiceUI(); return; }
        formData.append('file', fileInput.files[0]);
    }
    else if (currentVoiceInput === 'youtube') {
        const url = document.getElementById('youtube-url').value;
        if (!url) { alert("Please enter a URL."); resetVoiceUI(); return; }
        formData.append('youtube_url', url);
        if (loading) {
            const p = loading.querySelector('p');
            if (p) p.innerText = "Downloading audio from YouTube... (this takes time)";
        }
    }
    else if (currentVoiceInput === 'text') {
        const text = document.getElementById('transcript-text').value;
        if (!text) { alert("Please paste transcript."); resetVoiceUI(); return; }
        formData.append('transcript_text', text);
    }

    fetch('/api/analyze_voice', {
        method: 'POST',
        body: formData
    })
        .then(r => r.json())
        .then(res => {
            resetVoiceUI();
            if (btn) btn.innerText = "Re-Analyze";

            if (res.error) { alert("Error: " + res.error); return; }

            renderVoiceResults(res);
        })
        .catch(err => {
            console.error(err);
            resetVoiceUI();
            alert("Analysis failed. See console.");
        });
}

function resetVoiceUI() {
    const btn = document.querySelector('#step-4 .btn-secondary');
    if (btn) btn.classList.remove('hidden');

    const loading = document.getElementById('voice-loading');
    if (loading) {
        loading.classList.add('hidden');
        const p = loading.querySelector('p');
        if (p) p.innerText = "Analyzing micro-tremors and semantic patterns...";
    }
}

function renderVoiceResults(data) {
    const results = document.getElementById('voice-results');
    if (results) results.classList.remove('hidden');

    const btnNext5 = document.getElementById('btn-next-5');
    if (btnNext5) btnNext5.classList.remove('hidden');

    // 1. Micro-Tremors
    const tremorBar = document.getElementById('tremor-bar');
    const tremorDetail = document.getElementById('tremor-detail');

    if (tremorBar && tremorDetail) {
        if (data.audio_analysis && !data.audio_analysis.error && data.audio_analysis.tremor_score !== undefined) {
            const score = data.audio_analysis.tremor_score;
            const pct = (score * 100).toFixed(0);
            tremorBar.style.width = `${pct}%`;

            if (score < 0.3) {
                tremorBar.style.background = '#4ade80';
                tremorDetail.innerHTML = `✅ <strong>Low Stress (${pct}%)</strong>: Voice is steady.`;
            } else if (score < 0.6) {
                tremorBar.style.background = '#fbbf24';
                tremorDetail.innerHTML = `⚠️ <strong>Moderate Jitter (${pct}%)</strong>: Some hesitation.`;
            } else {
                tremorBar.style.background = '#f87171';
                tremorDetail.innerHTML = `🚨 <strong>High Stress (${pct}%)</strong>: Excessive jitter!`;
            }
        } else {
            tremorBar.style.width = '0%';
            tremorDetail.innerHTML = "N/A (Text Analysis Only)";
        }
    }

    // 2. Semantic Drift
    const semStatus = document.getElementById('semantic-status');
    const semDetail = document.getElementById('semantic-detail');

    if (semStatus && semDetail) {
        if (data.text_analysis && !data.text_analysis.error) {
            const risk = data.text_analysis.risk_score || 0;

            if (risk < 0.3) {
                semStatus.innerText = "CLEAR";
                semStatus.style.background = "#065f46"; semStatus.style.color = "#a7f3d0";
            } else if (risk < 0.7) {
                semStatus.innerText = "VAGUE";
                semStatus.style.background = "#92400e"; semStatus.style.color = "#fcd34d";
            } else {
                semStatus.innerText = "EVASIVE";
                semStatus.style.background = "#991b1b"; semStatus.style.color = "#fca5a5";
            }

            let report = `<strong>Risk Score: ${(risk * 100).toFixed(0)}%</strong><br>`;
            if (data.text_analysis.vagueness_check) report += `• ${data.text_analysis.vagueness_check}<br>`;
            if (data.text_analysis.evasion_check) report += `• ${data.text_analysis.evasion_check}<br>`;

            semDetail.innerHTML = report;
        } else {
            semStatus.innerText = "SKIPPED";
            semStatus.style.background = "#333";
            semDetail.innerText = "No semantic analysis available.";
        }
    }

    // 3. Transcription
    const transBox = document.getElementById('transcription-box');
    const transContent = document.getElementById('transcription-content-preview');
    if (data.transcription && transBox && transContent) {
        transBox.classList.remove('hidden');
        transContent.innerText = data.transcription.substring(0, 150);
    }

    // Auto-save voice results for fusion
    if (data.audio_analysis) fetchedMetadata.voice_audio = data.audio_analysis;
    if (data.text_analysis) fetchedMetadata.voice_text = data.text_analysis;
    saveState();
}

// ── Step 5: Fusion ────────────────────────────────────────────────────────

function goToStep(step) {
    // Override for Step 5 to trigger calculation
    if (step === 5) {
        runFusion();
    }

    // 1. Update Sidebar
    document.querySelectorAll('.sidebar li').forEach(li => {
        li.classList.remove('active');
        if (parseInt(li.dataset.step) === step) li.classList.add('active');
    });

    // 2. Show/Hide Steps (Inline Style Verification Fix)
    document.querySelectorAll('.step-card').forEach(card => {
        card.classList.add('hidden');
        card.classList.remove('active');
        card.style.display = 'none'; // Force hide
    });

    const target = document.getElementById(`step-${step}`);
    if (target) {
        target.classList.remove('hidden');
        target.style.display = 'block'; // Force show

        // Slight delay to allow display:block to kick in for transition
        setTimeout(() => {
            target.classList.add('active');
            // Resize map if entering step 2
            if (step === 2 && map) {
                map.invalidateSize();
            }
        }, 10);
    }

    // 3. Save State
    localStorage.setItem('step', step);
}

function runFusion() {
    const container = document.getElementById('step-5');
    if (!container) return;

    container.innerHTML = `
        <h2>Step 5: Fusion Verdict</h2>
        <div style="text-align:center; padding: 40px;">
            <div class="spinner"></div>
            <p>Aggregating signals & calculating CRDI™ score...</p>
        </div>
    `;

    // Prepare data modules
    const geoRisk = fetchedMetadata.geo_risk !== undefined ? fetchedMetadata.geo_risk : 0.5;

    // Construct payload
    const payload = {
        company: document.getElementById('company-name').value,
        geospatial_result: {
            avg_shell_risk: geoRisk,
            num_images: fetchedImages.length,
            verdict: geoRisk > 0.7 ? "High Risk" : "Low Risk"
        },
        behavioral_result: {
            combined_behavioral_score: calculateVoiceRisk(), // Helper
            audio_analysis: fetchedMetadata.voice_audio || {},
            semantic_analysis: fetchedMetadata.voice_text || {}
        }
    };

    fetch('/api/generate_score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
        .then(r => r.json())
        .then(data => {
            renderFusionReport(data);
        })
        .catch(e => {
            container.innerHTML = `<h2>Step 5: Fusion Verdict</h2><p style="color:red">Error generating report: ${e}</p>`;
        });
}

function calculateVoiceRisk() {
    let score = 0.5;
    if (fetchedMetadata.voice_audio && fetchedMetadata.voice_audio.tremor_score) {
        score = fetchedMetadata.voice_audio.tremor_score;
    }
    if (fetchedMetadata.voice_text && fetchedMetadata.voice_text.risk_score) {
        score = (score + fetchedMetadata.voice_text.risk_score) / 2;
    }
    return score;
}

function renderFusionReport(data) {
    const container = document.getElementById('step-5');

    let color = '#fbbf24';
    if (data.risk_label === 'CREDIBLE') color = '#4ade80';
    if (data.risk_label === 'SUSPICIOUS') color = '#fbbf24';
    if (data.risk_label === 'HIGH_RISK' || data.risk_label === 'CRITICAL') color = '#f87171';

    let html = `
        <h2>Step 5: Fusion Verdict</h2>
        <div style="text-align:center; padding: 2rem; border-bottom: 1px solid #333;">
            <div style="font-size: 3rem; margin-bottom: 10px;">${data.risk_icon}</div>
            <h1 style="font-size: 2.5rem; margin: 0;">CRDI Score: <span style="color:${color}">${(data.crdi_score * 100).toFixed(1)}%</span></h1>
            <p style="font-size: 1.2rem; color: #ccc; margin-top: 5px;">Verdict: <strong style="color:${color}">${data.risk_label}</strong></p>
            <p style="color: #888; font-style: italic;">"${data.risk_description}"</p>
        </div>

        <div class="results-grid" style="margin-top: 30px;">
    `;

    data.breakdown.forEach(item => {
        html += `
            <div class="metric-card" style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px;">
                <h3 style="margin: 0 0 15px 0; border-bottom: 1px solid #444; padding-bottom: 10px;">${item.module}</h3>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span>Contribution</span>
                    <strong>${(item.score * 100).toFixed(0)}%</strong>
                </div>
                <div style="width:100%; height:6px; background:#333; margin: 10px 0; border-radius:3px;">
                    <div style="width:${item.score * 100}%; height:100%; background:${color}; border-radius:3px;"></div>
                </div>
                <ul style="font-size: 0.9rem; color: #bbb; padding-left: 20px; margin-top: 10px;">
        `;

        // Dynamic details based on module
        if (item.module.includes("Geospatial")) {
            html += `<li>Images Analyzed: ${item.details.images_analyzed}</li>`;
            html += `<li>Max Risk Detect: ${(item.details.max_risk * 100).toFixed(0)}%</li>`;
        }
        if (item.module.includes("Behavioral")) {
            html += `<li>Audio Stress: ${(item.details.audio_stress * 100).toFixed(0)}%</li>`;
            html += `<li>Semantic Deception: ${(item.details.semantic_deception * 100).toFixed(0)}%</li>`;
        }

        html += `</ul></div>`;
    });

    html += `
        </div>
        <div class="actions" style="margin-top: 30px; text-align: center;">
            <button class="btn-primary" onclick="alert('Feature coming soon: Download PDF Report')">📄 Download Full Report</button>
            <button class="btn-secondary" onclick="goToStep(1)">Start New Investigation</button>
        </div>
    `;

    container.innerHTML = html;
}

function resetApp() {
    if (confirm("This will clear all data and start over. Are you sure?")) {
        localStorage.clear();
        window.location.reload();
    }
}

/**
 * Credit Decision ADK - Frontend SPA Controller
 * Handles Dashboard, Multi-step Wizard, and Application History
 */

let currentStep = 1;
const totalSteps = 4;
let charts = {};

document.addEventListener('DOMContentLoaded', () => {
    // Initialize App
    initApp();

    // Navigation listener
    window.addEventListener('hashchange', handleRouting);

    // Wizard buttons
    const nextBtn = document.getElementById('nextBtn');
    const prevBtn = document.getElementById('prevBtn');
    const loanWizard = document.getElementById('loanWizard');

    if (nextBtn) nextBtn.addEventListener('click', handleWizardNext);
    if (prevBtn) prevBtn.addEventListener('click', handleWizardPrev);
    if (loanWizard) loanWizard.addEventListener('submit', submitLoanApplication);

    // DTI Calculator Listeners
    const dtiInputs = ['loan_amount', 'annual_income', 'existing_debt'];
    dtiInputs.forEach(id => {
        const el = document.querySelector(`[name="${id}"]`);
        if (el) el.addEventListener('input', checkDTI);
    });
});

function checkDTI() {
    const loan = parseFloat(document.querySelector('[name="loan_amount"]').value) || 0;
    const income = parseFloat(document.querySelector('[name="annual_income"]').value) || 1;
    const debt = parseFloat(document.querySelector('[name="existing_debt"]').value) || 0;

    const monthlyIncome = income / 12;
    // Aggressive safety check: 15% estimated APR monthly payment
    const estPayment = (loan * 0.15) / 12;
    const totalMonthlyDebt = (debt / 12) + estPayment;

    // Avoid division by zero
    const dti = monthlyIncome > 0 ? (totalMonthlyDebt / monthlyIncome) * 100 : 0;

    const warningEl = document.getElementById('dtiWarning');
    const nextBtn = document.getElementById('nextBtn');

    if (dti > 50.0) {
        warningEl.style.display = 'block';
        document.getElementById('dtiValue').textContent = dti.toFixed(1);
        nextBtn.disabled = true;
        nextBtn.style.opacity = '0.5';
        nextBtn.style.cursor = 'not-allowed';

        // Calculate max viable loan for 50% DTI
        // 50% = (Debt + Pmt) / Income
        // 0.5 * Income = Debt + Pmt
        // Pmt = 0.5 * Income - Debt
        // (Loan * 0.15)/12 = (0.5 * Income/12) - (Debt/12)
        // Loan = ((0.5 * monthlyIncome) - (debt/12)) * 12 / 0.15

        const availableForLoan = (0.5 * monthlyIncome) - (debt / 12);
        if (availableForLoan > 0) {
            const maxLoan = (availableForLoan * 12) / 0.15;
            document.getElementById('maxViableLoan').textContent = '$' + Math.max(0, Math.floor(maxLoan)).toLocaleString();
        } else {
            document.getElementById('maxViableLoan').textContent = 'None (Debt too high)';
        }
    } else {
        warningEl.style.display = 'none';
        nextBtn.disabled = false;
        nextBtn.style.opacity = '1';
        nextBtn.style.cursor = 'pointer';
    }
}

function initApp() {
    handleRouting();
    checkSystemHealth();
    // System health polling
    setInterval(checkSystemHealth, 30000);
}

function handleRouting() {
    const hash = window.location.hash.replace('#', '') || 'dashboard';
    switchView(hash);
}

async function switchView(viewId) {
    // Hide all views
    document.querySelectorAll('.view').forEach(v => v.style.display = 'none');

    // Show target view
    const target = document.getElementById(`view-${viewId}`);
    if (target) {
        target.style.display = 'block';
        target.classList.add('active');
    } else {
        // Fallback to dashboard
        const dashboard = document.getElementById('view-dashboard');
        if (dashboard) {
            dashboard.style.display = 'block';
            dashboard.classList.add('active');
        }
        viewId = 'dashboard';
    }

    // Update nav links
    document.querySelectorAll('.nav-item').forEach(item => {
        const itemHash = item.getAttribute('href').replace('#', '');
        item.classList.toggle('active', itemHash === viewId);
    });

    // View-specific loading
    if (viewId === 'dashboard') loadDashboard();
    if (viewId === 'applications') loadApplications();

    window.scrollTo(0, 0);
}

// --- DASHBOARD MODULE ---
async function loadDashboard() {
    try {
        const response = await fetch('/api/dashboard/stats');
        const data = await response.json();

        if (data.success) {
            updateStatsUI(data.stats);
            renderDashboardCharts(data.stats);
        }
    } catch (err) {
        console.error("Dashboard load failed:", err);
    }
}

function updateStatsUI(stats) {
    const els = {
        total: document.getElementById('stats-total'),
        approval: document.getElementById('stats-approval'),
        risk: document.getElementById('stats-risk'),
        latency: document.getElementById('stats-latency')
    };

    if (els.total) els.total.textContent = stats.total_count;
    if (els.approval) els.approval.textContent = stats.approval_rate + '%';
    if (els.risk) els.risk.textContent = stats.avg_risk + '%';
    if (els.latency) els.latency.textContent = Math.round(stats.avg_latency) + 'ms';
}

function renderDashboardCharts(stats) {
    // Trend Chart (Line)
    const trendCtx = document.getElementById('trendChart');
    if (trendCtx) {
        if (charts.trend) charts.trend.destroy();
        charts.trend = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: stats.trends.map(t => t.date),
                datasets: [{
                    label: 'Applications',
                    data: stats.trends.map(t => t.count),
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    // Mix Chart (Doughnut)
    const mixCtx = document.getElementById('mixChart');
    if (mixCtx) {
        if (charts.mix) charts.mix.destroy();
        charts.mix = new Chart(mixCtx, {
            type: 'doughnut',
            data: {
                labels: ['Approve', 'Reject', 'Manual'],
                datasets: [{
                    data: [
                        stats.decision_dist.APPROVE || 0,
                        stats.decision_dist.REJECT || 0,
                        stats.decision_dist.MANUAL_REVIEW || 0
                    ],
                    backgroundColor: ['#10b981', '#f43f5e', '#f59e0b'],
                    borderWidth: 0
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, cutout: '70%' }
        });
    }
}

// --- WIZARD MODULE ---
function handleWizardNext() {
    if (validateStep(currentStep)) {
        if (currentStep < totalSteps) {
            currentStep++;
            updateWizardUI();
        }
    }
}

function handleWizardPrev() {
    if (currentStep > 1) {
        currentStep--;
        updateWizardUI();
    }
}

function updateWizardUI() {
    // Target step visibility
    document.querySelectorAll('.form-step').forEach(step => {
        step.classList.toggle('active', parseInt(step.dataset.step) === currentStep);
    });

    // Buttons
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');

    if (prevBtn) prevBtn.style.display = currentStep > 1 ? 'inline-flex' : 'none';
    if (nextBtn) nextBtn.style.display = currentStep < totalSteps ? 'inline-flex' : 'none';
    if (submitBtn) submitBtn.style.display = currentStep === totalSteps ? 'inline-flex' : 'none';

    // Progress bar
    const fill = document.getElementById('wizardFill');
    if (fill) fill.style.width = ((currentStep - 1) / (totalSteps - 1)) * 100 + '%';

    // Review Step Population
    if (currentStep === 4) {
        populateReview();
    }
}

function validateStep(step) {
    const activeStepEl = document.querySelector(`.form-step[data-step="${step}"]`);
    if (!activeStepEl) return true;

    const inputs = activeStepEl.querySelectorAll('input[required], select[required]');
    let valid = true;

    inputs.forEach(input => {
        if (!input.value) {
            input.style.borderColor = '#f43f5e';
            valid = false;
        } else {
            input.style.borderColor = 'var(--glass-border)';
        }
    });

    return valid;
}

function populateReview() {
    const form = document.getElementById('loanWizard');
    if (!form) return;

    const formData = new FormData(form);
    const summary = document.getElementById('reviewSummary');
    if (!summary) return;

    // Visual Cues Calculation
    const amt = parseFloat(formData.get('loan_amount') || 0);
    const inc = parseFloat(formData.get('annual_income') || 1);
    const score = parseInt(formData.get('credit_score') || 0);
    const dti = (parseFloat(formData.get('existing_debt') || 0) + (amt / 12)) / (inc / 12) * 100;

    const cues = {
        dti: dti < 15 ? 'excellent' : (dti > 45 ? 'warning' : 'stable'),
        score: score >= 700 ? 'excellent' : (score < 600 ? 'warning' : 'stable'),
        loan: (amt / inc) < 0.2 ? 'excellent' : 'stable'
    };

    let html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; text-align: left;">';

    const getIcon = (type) => {
        if (type === 'excellent') return '<i class="fas fa-check-circle cue-excellent" title="Positive Indicator"></i>';
        if (type === 'warning') return '<i class="fas fa-exclamation-triangle cue-warning" title="Risk Warning"></i>';
        return '<i class="fas fa-info-circle cue-stable" title="Neutral Indicator"></i>';
    };

    for (let [key, value] of formData.entries()) {
        const label = key.replace(/_/g, ' ').toUpperCase();
        let cueIcon = '';

        if (key === 'credit_score') cueIcon = getIcon(cues.score);
        if (key === 'annual_income' || key === 'existing_debt') cueIcon = getIcon(cues.dti);
        if (key === 'loan_amount') cueIcon = getIcon(cues.loan);

        html += `<div class="review-item">
            <div>
                <small class="text-muted">${label}</small>
                <p>${cueIcon} ${value}</p>
            </div>
        </div>`;
    }

    html += '</div>';
    html += `<div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 8px;">
        <small class="text-muted">PRE-SCREENING INSIGHT:</small>
        <p>Your Estimated Debt-to-Income (DTI) is <span class="cue-${cues.dti}" style="font-weight: bold;">${dti.toFixed(1)}%</span>. 
        ${dti < 10 ? 'This is exceptionally strong.' : (dti > 40 ? 'This may impact the final decision.' : 'This is within normal ranges.')}</p>
    </div>`;

    summary.innerHTML = html;
}

async function submitLoanApplication(e) {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const timeline = document.getElementById('processingTimeline');
    const wizardResults = document.getElementById('wizardResults');
    const formStep = document.querySelector('.form-step[data-step="4"]');

    btn.disabled = true;
    formStep.style.opacity = '0.5';
    timeline.style.display = 'block';
    wizardResults.style.display = 'none';

    // Timeline Simulation
    const runTimeline = async () => {
        const steps = [
            { id: 't-step-1', duration: 1500 },
            { id: 't-step-2', duration: 4000 },
            { id: 't-step-3', duration: 3000 },
            { id: 't-step-4', duration: 2000 }
        ];

        for (const step of steps) {
            const el = document.getElementById(step.id);
            if (!el) continue;

            el.classList.add('active');
            el.querySelector('i').className = 'fas fa-spinner fa-spin';

            const start = Date.now();
            await new Promise(r => setTimeout(r, step.duration));
            const end = Date.now();

            el.classList.remove('active');
            el.classList.add('complete');
            el.querySelector('i').className = 'fas fa-check-circle';
            el.querySelector('.t-time').textContent = `${((end - start) / 1000).toFixed(1)}s`;
        }
    };

    const formData = new FormData(e.target);
    const payload = Object.fromEntries(formData.entries());

    // Run both in parallel
    const [timelineRes, apiRes] = await Promise.all([
        runTimeline(),
        fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).then(r => r.json())
    ]);

    timeline.style.display = 'none';
    formStep.style.opacity = '1';
    btn.disabled = false;

    if (apiRes.success) {
        displayWizardResults(apiRes.result, payload);
    } else {
        alert("Evaluation Pipeline Error: " + apiRes.error);
    }
}

function displayWizardResults(result, originalInputs) {
    const container = document.getElementById('wizardResults');
    if (!container) return;

    const decisionColor = result.decision === 'APPROVE' ? 'var(--success)' : (result.decision === 'REJECT' ? 'var(--error)' : 'var(--warning)');
    const riskBarsHtml = renderRiskBars(result.component_scores || {});

    container.innerHTML = `
        <div class="card" style="border-top: 4px solid ${decisionColor}; animation: fadeIn 0.8s ease-out;">
            <div class="result-header" style="text-align: center;">
                <div class="result-score-circle" style="border-color: ${decisionColor}; margin: 0 auto 1.5rem;">
                    <span class="val" style="font-size: 2.5rem; display: block;">${Math.round(result.risk_score * 100)}%</span>
                    <span class="label" style="font-size: 0.8rem; color: var(--text-muted);">RISK SCORE</span>
                </div>
                <h2 style="color: ${decisionColor}; font-size: 2.5rem; letter-spacing: 2px;">${result.decision}</h2>
                <p class="text-muted">Analysis Confidence: ${Math.round(result.confidence * 100)}%</p>
            </div>

            <div class="grid-2" style="margin-top: 3rem; gap: 2rem;">
                <div>
                     <h3 style="font-size: 1.1rem; margin-bottom: 1.5rem;"><i class="fas fa-layer-group"></i> Risk Dimensions Analysis</h3>
                     <div class="risk-bar-container">
                        ${riskBarsHtml}
                     </div>
                </div>
                <div class="card" style="background: rgba(255,255,255,0.03); border: 1px solid var(--glass-border);">
                    <h3 style="font-size: 1.1rem; margin-bottom: 1rem;"><i class="fas fa-robot"></i> AI Assessment Narrative</h3>
                    <p style="font-size: 0.95rem; line-height: 1.6;">${result.explanation}</p>
                </div>
            </div>

            <div class="what-if-panel" style="margin-top: 3rem;">
                <h3 style="font-size: 1.1rem;"><i class="fas fa-adjust"></i> "What If" Decision Simulation</h3>
                <p class="text-muted" style="font-size: 0.85rem; margin-bottom: 1.5rem;">Adjust the credit score to see how the policy rules would respond to this specific financial profile.</p>
                
                <div class="slider-group">
                    <label style="margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                        <span>Simulated Credit Score</span> 
                        <span id="val-score" style="font-weight: bold; color: var(--primary); font-size: 1.2rem;">${originalInputs.credit_score}</span>
                    </label>
                    <input type="range" id="slider-score" min="300" max="850" value="${originalInputs.credit_score}" style="width: 100%; cursor: pointer;">
                </div>
                
                <div id="what-if-result" style="margin-top: 1.5rem; padding: 1.5rem; border-radius: 12px; background: rgba(0,0,0,0.3); text-align: center; border: 1px solid var(--glass-border);">
                    <small class="text-muted" style="text-transform: uppercase; letter-spacing: 1px;">Hypothetical Policy Outcome</small>
                    <div id="hypo-decision" style="font-weight: 800; font-size: 1.8rem; color: ${decisionColor}; transition: all 0.3s ease;">${result.decision}</div>
                </div>
            </div>

            <div style="display:flex; justify-content: center; gap: 1.5rem; margin-top: 3rem;">
                <button class="btn btn-primary" onclick="window.location.hash='#applications'">
                    <i class="fas fa-list-ul"></i> Audit Trail
                </button>
                <button class="btn btn-secondary" onclick="resetWizard()">
                    <i class="fas fa-redo"></i> Start New Analysis
                </button>
            </div>
        </div>
    `;
    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });

    initWhatIfLogic(result, originalInputs);
}

function renderRiskBars(scores) {
    const dims = Object.keys(scores);
    if (dims.length === 0) return '<p class="text-muted">Detailed dimension scores unavailable for this record.</p>';

    return dims.map(dim => {
        const val = (scores[dim].overall_default_rate || 0) * 100;
        let color = 'var(--success)';
        if (val > 25) color = 'var(--warning)';
        if (val > 50) color = 'var(--error)';

        return `
            <div class="risk-bar-row">
                <div class="risk-bar-label">
                    <span>${dim.replace(/_/g, ' ').toUpperCase()}</span>
                    <span style="font-weight: bold;">${Math.round(val)}%</span>
                </div>
                <div class="risk-bar-bg"><div class="risk-bar-fill" style="width: ${val}%; background: ${color}"></div></div>
            </div>
        `;
    }).join('');
}

function initWhatIfLogic(currentResult, originalInputs) {
    const slider = document.getElementById('slider-score');
    const valDisplay = document.getElementById('val-score');
    const hypoDecision = document.getElementById('hypo-decision');

    slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        valDisplay.textContent = val;

        const amt = parseFloat(originalInputs.loan_amount);
        const inc = parseFloat(originalInputs.annual_income);
        const debt = parseFloat(originalInputs.existing_debt || 0);
        const dti = (debt + (amt / 12)) / (inc / 12) * 100;
        const years = parseFloat(originalInputs.years_employed || 1);

        // Simulated credit penalty (matches backend mapping)
        let creditPenalty = 0.0;
        if (val < 750) {
            if (val >= 700) creditPenalty = 0.05;
            else if (val >= 650) creditPenalty = 0.12;
            else if (val >= 600) creditPenalty = 0.20;
            else if (val >= 550) creditPenalty = 0.35;
            else creditPenalty = 0.55;
        }

        // Weighted Risk Calculation (Simplified Backend Logic)
        const baseRisk = (0.2 * 0.65) + (creditPenalty * 0.35); // Assuming 0.2 base default rate
        const adjustedRisk = Math.min(1.0, baseRisk);

        let decision = "APPROVE";
        let color = 'var(--success)';

        if (dti > 65 || adjustedRisk > 0.40) {
            decision = "REJECT";
            color = 'var(--error)';
        } else if (adjustedRisk > 0.15 || years < 0.3 || val < 580) {
            decision = "MANUAL_REVIEW";
            color = 'var(--warning)';
        }

        hypoDecision.textContent = decision;
        hypoDecision.style.color = color;
    });
}

// --- HISTORY MODULE ---
async function loadApplications() {
    try {
        const response = await fetch('/api/applications');
        const data = await response.json();
        const body = document.getElementById('applicationsBody');
        if (!body) return;
        body.innerHTML = '';

        if (!data.applications || data.applications.length === 0) {
            body.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 4rem; color: var(--text-muted);">No applications found in the system.</td></tr>';
            return;
        }

        data.applications.forEach(app => {
            const date = new Date(app.timestamp).toLocaleString();
            const badgeClass = `badge-${app.decision.toLowerCase().replace(/ /g, '-')}`;

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${date}</td>
                <td><code style="background: rgba(255,255,255,0.05); padding: 2px 4px; border-radius: 4px;">${app.decision_id.substring(0, 8)}</code></td>
                <td><span style="font-weight: bold;">${Math.round(app.risk_score * 100)}%</span></td>
                <td><span class="badge ${badgeClass}">${app.decision}</span></td>
                <td>${Math.round(app.confidence * 100)}%</td>
                <td>
                    <button class="btn btn-secondary" onclick="loadApplicationDetail('${app.decision_id}')" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">
                        <i class="fas fa-search-plus"></i> Trace
                    </button>
                </td>
            `;
            body.appendChild(tr);
        });
    } catch (err) {
        console.error("Audit log loading failed:", err);
    }
}

async function loadApplicationDetail(id) {
    switchView('details');
    const container = document.getElementById('applicationDetailContent');
    if (!container) return;
    container.innerHTML = '<div style="text-align: center; padding: 6rem;"><div class="btn-loader" style="width: 40px; height: 40px;"></div><p style="margin-top: 1rem; color: var(--text-muted);">Decrypting audit trail data...</p></div>';

    try {
        const response = await fetch(`/api/applications/${id}`);
        const data = await response.json();
        if (data.success) {
            renderDetailView(data.application, container);
        } else {
            container.innerHTML = `<div class="card" style="border-left: 4px solid var(--error);"><h2 class="text-error">Audit Retrieval Failed</h2><p>${data.error}</p></div>`;
        }
    } catch (err) {
        container.innerHTML = '<div class="card" style="border-left: 4px solid var(--error);"><h2>Connectivity Error</h2><p>Could not reach the audit engine.</p></div>';
    }
}

function renderDetailView(app, container) {
    const decisionColor = app.decision === 'APPROVE' ? 'var(--success)' : (app.decision === 'REJECT' ? 'var(--error)' : 'var(--warning)');
    const badgeClass = `badge-${app.decision.toLowerCase().replace(/ /g, '-')}`;

    // Extract raw inputs if available
    const rawInputs = app.inputs_json ? (typeof app.inputs_json === 'string' ? JSON.parse(app.inputs_json) : app.inputs_json) : {};

    container.innerHTML = `
        <div class="card" style="border-left: 4px solid ${decisionColor}; animation: fadeIn 0.5s ease-out; padding: 2.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 3rem; flex-wrap: wrap; gap: 1.5rem;">
                <div>
                    <h1 style="font-size: 2rem; margin-bottom: 0.5rem; letter-spacing: -0.5px;">Audit Trail Analysis</h1>
                    <p class="text-muted" style="font-family: monospace; font-size: 0.95rem;">REF_ID: ${app.decision_id}</p>
                    <p class="text-muted"><i class="far fa-clock"></i> Captured: ${new Date(app.timestamp).toLocaleString()}</p>
                </div>
                <div class="badge ${badgeClass}" style="font-size: 1.5rem; padding: 0.8rem 2.5rem; box-shadow: var(--shadow-premium); border-radius: var(--radius-md);">
                    ${app.decision}
                </div>
            </div>

            <div class="grid-2" style="gap: 4rem;">
                <div class="detail-main">
                    <section style="margin-bottom: 3rem;">
                        <h3 style="font-size: 1.2rem; margin-bottom: 1.25rem; color: var(--text-main); display: flex; align-items: center; gap: 0.75rem;">
                            <i class="fas fa-quote-left" style="color: var(--primary); font-size: 0.9rem;"></i> 
                            AI Decision Rationale
                        </h3>
                        <div class="card" style="background: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); padding: 1.5rem; line-height: 1.8;">
                            ${app.explanation_summary}
                        </div>
                    </section>
                    
                    <section>
                        <h3 style="font-size: 1.2rem; margin-bottom: 1.5rem; color: var(--text-main); display: flex; align-items: center; gap: 0.75rem;">
                            <i class="fas fa-chart-bar" style="color: var(--primary);"></i> 
                            Dimensional Risk Profile
                        </h3>
                        <div class="risk-bar-container" style="padding: 0 0.5rem;">
                            ${renderRiskBars(app.component_scores || {})}
                        </div>
                    </section>
                </div>

                <div class="detail-sidebar">
                    <div class="card" style="background: var(--sidebar-bg); border: 1px solid var(--glass-border); padding: 2rem; margin-bottom: 2rem; box-shadow: var(--shadow-premium);">
                        <h3 style="font-size: 1.1rem; margin-bottom: 2rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">Core Probability Metrics</h3>
                        
                        <div style="margin-bottom: 2.5rem;">
                            <small class="text-muted" style="display: block; margin-bottom: 0.5rem; font-weight: 600;">ADJUSTED RISK EXPOSURE</small>
                            <h2 style="color: ${decisionColor}; font-size: 3.5rem; font-weight: 800; line-height: 1;">${Math.round(app.risk_score * 100)}%</h2>
                        </div>
                        
                        <div style="margin-bottom: 2.5rem;">
                            <small class="text-muted" style="display: block; margin-bottom: 0.5rem; font-weight: 600;">MODEL CONFIDENCE</small>
                            <h2 style="font-size: 2rem; font-weight: 700;">${Math.round(app.confidence * 100)}%</h2>
                        </div>
                        
                        <div>
                            <small class="text-muted" style="display: block; margin-bottom: 0.5rem; font-weight: 600;">EVALUATION LATENCY</small>
                            <p style="font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; color: var(--primary);">${Math.round(app.latency_ms)}ms</p>
                        </div>
                    </div>

                    <!-- Technical Audit Section -->
                    <div class="card" style="background: rgba(0,0,0,0.15); border: 1px dashed var(--glass-border); padding: 1.5rem;">
                        <h4 style="font-size: 0.9rem; margin-bottom: 1rem; color: var(--text-muted);"><i class="fas fa-code"></i> TECHNICAL AUDIT (RAW INPUTS)</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem;">
                            ${Object.entries(rawInputs).map(([k, v]) => `
                                <div>
                                    <span class="text-muted" style="font-size: 0.7rem; text-transform: uppercase; display: block;">${k}</span>
                                    <span style="font-family: monospace;">${v}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <button class="btn btn-secondary" onclick="switchView('applications')" style="width: 100%; margin-top: 2rem; padding: 1rem; font-weight: 600;">
                        <i class="fas fa-chevron-left"></i> Return to Audit History
                    </button>
                </div>
            </div>
        </div>
    `;
}

// --- UTILS ---
async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        const text = document.getElementById('statusText');
        if (text) {
            text.textContent = data.status === 'healthy' ? 'ONLINE' : 'DEGRADED';
            text.style.color = data.status === 'healthy' ? 'var(--success)' : 'var(--warning)';
        }
    } catch (err) {
        const text = document.getElementById('statusText');
        if (text) {
            text.textContent = 'OFFLINE';
            text.style.color = 'var(--error)';
        }
    }
}

function resetWizard() {
    const form = document.getElementById('loanWizard');
    if (form) form.reset();
    currentStep = 1;
    updateWizardUI();
    const results = document.getElementById('wizardResults');
    if (results) results.style.display = 'none';
    const timeline = document.getElementById('processingTimeline');
    if (timeline) {
        timeline.style.display = 'none';
        document.querySelectorAll('.t-step').forEach(s => {
            s.classList.remove('active', 'complete');
            s.querySelector('i').className = 'far fa-circle';
            s.querySelector('.t-time').textContent = '';
        });
    }
    window.scrollTo(0, 0);
}

// Global Demo Data Trigger
window.fillDemoData = () => {
    const data = {
        applicant_name: 'Dhia Architecture',
        state: 'NY',
        homeownership: 'OWN',
        loan_amount: 5000,
        annual_income: 90000,
        existing_debt: 1200,
        loan_purpose: 'major_purchase',
        employment_status: 'Senior Architect',
        years_employed: 3,
        credit_score: 599
    };

    for (let key in data) {
        const el = document.querySelector(`[name="${key}"]`);
        if (el) el.value = data[key];
    }
    currentStep = 4;
    updateWizardUI();
};
import { useEffect, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

const algorithms = [
  { id: 'linear_regression', name: 'Linear Regression', short: 'LR', category: 'Prediction', description: 'Estimate an illustrative loan amount from financial inputs.', endpoint: '/predict/loan-estimate', fields: ['income', 'debt', 'credit_score', 'history'] },
  { id: 'naive_bayes', name: 'Naive Bayes', short: 'NB', category: 'Classification', description: 'Classify a message as ordinary or potentially suspicious.', endpoint: '/predict/message', fields: ['message'] },
  { id: 'knn', name: 'K-Nearest Neighbors', short: 'KNN', category: 'Classification', description: 'Classify a profile using nearby examples in feature space.', endpoint: '/predict/knn', fields: ['feature_1', 'feature_2', 'feature_3', 'feature_4'] },
  { id: 'svm', name: 'Support Vector Machine', short: 'SVM', category: 'Classification', description: 'Separate synthetic classes using a learned decision boundary.', endpoint: '/predict/credit-risk', fields: ['feature_1', 'feature_2', 'feature_3', 'feature_4'] },
  { id: 'decision_tree', name: 'Decision Tree', short: 'DT', category: 'Classification', description: 'Classify an example using learned decision rules.', endpoint: '/predict/loan-decision', fields: ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'] },
  { id: 'ann', name: 'Artificial Neural Network', short: 'ANN', category: 'Classification', description: 'Classify a synthetic transaction example using a neural network.', endpoint: '/predict/fraud', fields: ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'] },
]

const initialValues = {
  income: '60000',
  debt: '10000',
  credit_score: '700',
  history: '5',
  message: 'Urgent click this link to claim your prize',
  feature_1: '0.5',
  feature_2: '-0.3',
  feature_3: '1.2',
  feature_4: '0.8',
  feature_5: '0.4',
}

const labels = {
  income: 'Annual income (PHP)',
  debt: 'Existing debt (PHP)',
  credit_score: 'Credit score (300–850)',
  history: 'Credit history (years)',
  message: 'Message to analyze',
  feature_1: 'Feature 1',
  feature_2: 'Feature 2',
  feature_3: 'Feature 3',
  feature_4: 'Feature 4',
  feature_5: 'Feature 5',
}

function App() {
  const [selectedId, setSelectedId] = useState('linear_regression')
  const [values, setValues] = useState(initialValues)
  const [result, setResult] = useState(null)
  const [metrics, setMetrics] = useState({})
  const [apiStatus, setApiStatus] = useState('checking')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const selected = algorithms.find((item) => item.id === selectedId)

  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then((response) => {
        if (!response.ok) throw new Error('API unavailable')
        return response.json()
      })
      .then((data) => {
        const nextMetrics = {}
        data.models?.forEach((model) => {
          nextMetrics[model.name] = model.evaluation
        })
        setMetrics(nextMetrics)
        setApiStatus('online')
      })
      .catch(() => setApiStatus('offline'))
  }, [])

  function chooseAlgorithm(id) {
    setSelectedId(id)
    setResult(null)
    setError('')
  }

  function updateValue(event) {
    setValues((previous) => ({
      ...previous,
      [event.target.name]: event.target.value,
    }))
  }

  async function runPrediction(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    const payload = {}
    selected.fields.forEach((field) => {
      payload[field] = field === 'message'
        ? values[field]
        : Number(values[field])
    })

    try {
      const response = await fetch(`${API_URL}${selected.endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(', ')
          : data.detail
        throw new Error(detail || 'The prediction request failed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Could not connect to the API. Make sure your FastAPI server is running at http://127.0.0.1:8000.'
          : err.message,
      )
    } finally {
      setLoading(false)
    }
  }

  function metricText(algorithm) {
    const modelMetrics = metrics[algorithm.name]
    if (!modelMetrics) return 'Model trained'
    if (modelMetrics.accuracy !== undefined) {
      return `${(modelMetrics.accuracy * 100).toFixed(1)}% accuracy`
    }
    if (modelMetrics.mae !== undefined) {
      return `₱${Number(modelMetrics.mae).toLocaleString()} MAE`
    }
    return 'Evaluation available'
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview">
          <span className="brand-mark">A</span>
          <span>
            <strong>Aegis</strong>
            <small>CREDIT SYSTEMS</small>
          </span>
        </a>

        <p className="nav-label">WORKSPACE</p>
        <a className="nav-link active" href="#overview"><span>▦</span> Overview</a>
        <a className="nav-link" href="#algorithms"><span>⌘</span> Algorithms</a>
        <a className="nav-link" href="#playground"><span>⌁</span> Prediction lab</a>

        <div className="sidebar-bottom">
          <div className="secure-icon">✳</div>
          <strong>Learning securely</strong>
          <p>Exploring machine learning through synthetic data.</p>
          <span className="demo-tag">ACADEMIC DEMO</span>
        </div>
        <div className="sidebar-footer">AEGIS CREDIT · 2026</div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Overview</div>
          <div className={`api-indicator ${apiStatus}`}>
            <span className="status-dot" />
            {apiStatus === 'checking' ? 'Connecting to API' : apiStatus === 'online' ? 'API connected' : 'API disconnected'}
          </div>
        </header>

        <section className="welcome-section">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> MACHINE LEARNING WORKSPACE</div>
            <h1>Understand risk.<br /><span>Explore intelligence.</span></h1>
            <p className="welcome-copy">
              Explore six machine-learning algorithms through interactive
              financial technology demonstrations.
            </p>
            <a className="primary-link" href="#playground">Explore prediction lab <span>↗</span></a>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="hero-core"><span>✳</span></div>
            <div className="float-chip chip-top">ML <span>ENGINE</span></div>
            <div className="float-chip chip-bottom"><span className="status-dot" /> MODEL SYSTEMS ONLINE</div>
            <div className="hero-line line-one" />
            <div className="hero-line line-two" />
          </div>
        </section>

        <section className="stats-grid" aria-label="Project overview">
          <article className="stat-card">
            <div className="stat-icon">⌘</div>
            <span className="stat-label">ALGORITHMS</span>
            <strong>06</strong>
            <small>Core ML techniques</small>
          </article>
          <article className="stat-card">
            <div className="stat-icon">↗</div>
            <span className="stat-label">API STATUS</span>
            <strong className={apiStatus === 'online' ? 'text-green' : ''}>{apiStatus === 'online' ? 'Online' : apiStatus === 'checking' ? 'Checking' : 'Offline'}</strong>
            <small>FastAPI prediction service</small>
          </article>
          <article className="stat-card">
            <div className="stat-icon">◈</div>
            <span className="stat-label">DATA SOURCE</span>
            <strong>Synthetic</strong>
            <small>Generated demo datasets</small>
          </article>
          <article className="stat-card">
            <div className="stat-icon">✓</div>
            <span className="stat-label">API TESTS</span>
            <strong>6 / 6</strong>
            <small>Prediction endpoints tested</small>
          </article>
        </section>

        <section className="section-block" id="algorithms">
          <div className="section-heading">
            <div>
              <div className="eyebrow">THE TOOLKIT</div>
              <h2>Machine learning algorithms</h2>
              <p>Choose an algorithm to learn about it and try a prediction.</p>
            </div>
            <span className="section-count">06 MODELS</span>
          </div>

          <div className="algorithm-grid">
            {algorithms.map((algorithm, index) => (
              <button
                className={`algorithm-card ${selectedId === algorithm.id ? 'selected' : ''}`}
                key={algorithm.id}
                onClick={() => {
                  chooseAlgorithm(algorithm.id)
                  document.getElementById('playground')?.scrollIntoView({ behavior: 'smooth' })
                }}
              >
                <div className="algorithm-card-top">
                  <span className={`algorithm-symbol symbol-${index}`}>{algorithm.short}</span>
                  <span className="card-arrow">↗</span>
                </div>
                <span className="algorithm-category">{algorithm.category}</span>
                <h3>{algorithm.name}</h3>
                <p>{algorithm.description}</p>
                <div className="algorithm-card-footer">
                  <span>{metricText(algorithm)}</span>
                  <span className="try-label">Try model →</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="playground-section" id="playground">
          <div className="section-heading">
            <div>
              <div className="eyebrow">INTERACTIVE WORKSPACE</div>
              <h2>Prediction lab</h2>
              <p>Enter sample values and send them to the selected model.</p>
            </div>
            <span className="live-tag"><span className="status-dot" /> LIVE API</span>
          </div>

          <div className="lab-layout">
            <div className="lab-form-card">
              <div className="lab-card-heading">
                <div className="selected-model-icon">{selected.short}</div>
                <div>
                  <span className="algorithm-category">{selected.category}</span>
                  <h3>{selected.name}</h3>
                </div>
              </div>
              <p className="lab-description">{selected.description}</p>

              <form onSubmit={runPrediction}>
                <div className="field-grid">
                  {selected.fields.map((field) => (
                    <label className={`form-field ${field === 'message' ? 'full-field' : ''}`} key={field}>
                      <span>{labels[field]}</span>
                      {field === 'message' ? (
                        <textarea
                          name={field}
                          value={values[field]}
                          onChange={updateValue}
                          rows={4}
                          required
                          maxLength={5000}
                        />
                      ) : (
                        <input
                          name={field}
                          type="number"
                          value={values[field]}
                          onChange={updateValue}
                          step="any"
                          required
                        />
                      )}
                    </label>
                  ))}
                </div>
                <div className="form-note"><span>ⓘ</span> {selected.id === 'linear_regression' ? 'Amount estimates are illustrative and not loan offers.' : selected.id === 'naive_bayes' ? 'Never rely on this demo alone to identify scams.' : 'Feature values are synthetic and do not represent real financial data.'}</div>
                <button className="run-button" type="submit" disabled={loading || apiStatus === 'offline'}>
                  {loading ? 'Running prediction…' : 'Run prediction'} <span>→</span>
                </button>
                {apiStatus === 'offline' && <p className="error-message">Start your FastAPI backend, then refresh this page.</p>}
                {error && <p className="error-message">{error}</p>}
              </form>
            </div>

            <div className="result-card">
              <div className="result-topline"><span>MODEL OUTPUT</span><span className="result-dot" /></div>
              {!result ? (
                <div className="empty-result">
                  <div className="empty-result-icon">⌁</div>
                  <h3>Awaiting prediction</h3>
                  <p>Your model output will appear here after you submit the form.</p>
                  <div className="result-placeholder"><span /> <span /> <span /></div>
                </div>
              ) : (
                <div className="prediction-result">
                  <div className="success-mark">✓</div>
                  <span className="result-kicker">PREDICTION COMPLETE</span>
                  {result.estimated_amount !== undefined ? (
                    <>
                      <h3 className="amount-result">₱{Number(result.estimated_amount).toLocaleString('en-PH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</h3>
                      <p>Illustrative estimated loan amount</p>
                    </>
                  ) : (
                    <>
                      <h3>{result.classification || result.class_label || `Class ${result.predicted_class}`}</h3>
                      <p>Predicted class: {result.predicted_class}</p>
                    </>
                  )}
                  {result.class_probabilities && (
                    <div className="probability-list">
                      <strong>Class probabilities</strong>
                      {Object.entries(result.class_probabilities).map(([label, probability]) => (
                        <div className="probability-row" key={label}>
                          <div><span>Class {label}</span><span>{(probability * 100).toFixed(1)}%</span></div>
                          <div className="probability-track"><span style={{ width: `${probability * 100}%` }} /></div>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="result-disclaimer">{result.note}</p>
                  <button className="reset-button" onClick={() => setResult(null)}>Clear result</button>
                </div>
              )}
              <div className="result-footer"><span>AEGIS MODEL SERVICE</span><span>DEMO MODE</span></div>
            </div>
          </div>
        </section>

        <section className="notice-banner">
          <div className="notice-symbol">i</div>
          <div>
            <strong>Educational demonstration only</strong>
            <p>These models use synthetic data. Their metrics and predictions are not evidence of real-world performance and must not be used for lending, credit scoring, or fraud decisions.</p>
          </div>
        </section>

        <footer className="page-footer">
          <span>© 2026 Aegis Credit · Academic project</span>
          <span>Built for machine learning education <span className="footer-star">✳</span></span>
        </footer>
      </main>
    </div>
  )
}

export default App
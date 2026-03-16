import React, { useState } from 'react';

const fields = [
    'scaled_time',
    'scaled_amount',
    ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`)
];

const defaultState = fields.reduce((acc, field) => {
    acc[field] = '0.0';
    return acc;
}, {});

// Typical parameters for a non-fraud transaction based on standard scaling
const legitSample = fields.reduce((acc, field) => {
    acc[field] = (Math.random() * 0.5 - 0.25).toFixed(4);
    return acc;
}, {});
legitSample.scaled_amount = '1.2500';
legitSample.scaled_time = '-0.1500';

// Fraud transactions often have extreme outliers in V fields
const fraudSample = fields.reduce((acc, field) => {
    acc[field] = (Math.random() * 10 - 5).toFixed(4); // Larger variance
    return acc;
}, {});
fraudSample.scaled_amount = '15.5000';
fraudSample.scaled_time = '0.9900';
fraudSample.V4 = '4.5';
fraudSample.V11 = '3.5';
fraudSample.V12 = '-5.5';
fraudSample.V14 = '-6.5';
fraudSample.V17 = '-4.0';

export default function App() {
    const [formData, setFormData] = useState(defaultState);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleFillLegit = () => setFormData({ ...legitSample });
    const handleFillFraud = () => setFormData({ ...fraudSample });
    const handleClear = () => {
        setFormData({ ...defaultState });
        setResult(null);
        setError(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        // Convert strings to floats
        const payload = {};
        for (const key of fields) {
            payload[key] = parseFloat(formData[key]) || 0.0;
        }

        try {
            // With vite proxy pointing to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            <header className="header">
                <div className="logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0110 0v4"></path>
                    </svg>
                    SecureBank Vault
                </div>
            </header>

            <div className="card">
                <h2 className="card-title">Fraud Detection System</h2>
                <p className="card-subtitle">
                    Input the transaction parameters derived from the model below. Use the sample buttons to quickly populate the required fields.
                </p>

                <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                    <button type="button" className="btn" style={{ background: 'var(--primary)', width: 'auto', marginTop: 0 }} onClick={handleFillLegit}>
                        Load Normal Transaction
                    </button>
                    <button type="button" className="btn" style={{ background: '#ef4444', width: 'auto', marginTop: 0 }} onClick={handleFillFraud}>
                        Load Suspicious Transaction
                    </button>
                    <button type="button" className="btn" style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-main)', width: 'auto', marginTop: 0 }} onClick={handleClear}>
                        Clear
                    </button>
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="form-grid">
                        {fields.map(field => (
                            <div className="form-group" key={field}>
                                <label htmlFor={field}>{field}</label>
                                <input
                                    type="number"
                                    step="any"
                                    id={field}
                                    name={field}
                                    value={formData[field]}
                                    onChange={handleChange}
                                    required
                                />
                            </div>
                        ))}
                    </div>

                    <button type="submit" className="btn" disabled={loading}>
                        {loading ? 'Analyzing...' : 'Run Fraud Analysis'}
                    </button>
                </form>

                {error && (
                    <div className="result-box" style={{ background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}>
                        <p><strong>Error:</strong> {error}</p>
                        <p style={{ fontSize: '0.875rem', marginTop: '0.5rem', opacity: 0.8 }}>Please ensure the backend is running on port 8000.</p>
                    </div>
                )}

                {result && (
                    <div className={`result-box ${result.fraud_prediction === 1 ? 'fraud' : 'safe'}`}>
                        <div className="result-icon">
                            {result.fraud_prediction === 1 ? '🚨' : '✅'}
                        </div>
                        <h3 className="result-title">
                            {result.fraud_prediction === 1 ? 'Fraudulent Transaction Detected' : 'Transaction is Legitimate'}
                        </h3>
                        <p className="result-desc">
                            Fraud Probability: {(result.fraud_probability * 100).toFixed(2)}%
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}

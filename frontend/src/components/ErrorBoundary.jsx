import React from 'react';
import { AlertTriangle, RefreshCcw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an unhandled error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            margin: '2rem 0',
            padding: '2rem',
            background: 'rgba(244, 63, 94, 0.08)',
            border: '1px solid #f43f5e',
            borderRadius: '1rem',
            color: '#f8fafc',
            textAlign: 'center',
          }}
        >
          <div style={{ display: 'inline-flex', padding: '0.75rem', borderRadius: '50%', background: 'rgba(244, 63, 94, 0.2)', marginBottom: '1rem' }}>
            <AlertTriangle size={32} color="#fb7185" />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#fb7185' }}>
            Component Render Error
          </h3>
          <p style={{ color: '#cbd5e1', fontSize: '0.9rem', maxWidth: '600px', margin: '0 auto 1.5rem auto' }}>
            {this.state.error ? this.state.error.message : 'An unexpected rendering error occurred.'}
          </p>
          <button
            className="btn btn-primary"
            onClick={this.handleReset}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.25rem' }}
          >
            <RefreshCcw size={16} /> Recover View
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

import React from 'react';

export default function Stats() {
  return (
    <section style={{padding: '4rem 0', background: 'var(--ifm-background-surface-color)'}}>
      <div className="container">
        <div className="stats">
          <div className="stat">
            <div className="stat__number">10K+</div>
            <div className="stat__label">Downloads</div>
          </div>
          <div className="stat">
            <div className="stat__number">500+</div>
            <div className="stat__label">GitHub Stars</div>
          </div>
          <div className="stat">
            <div className="stat__number">3</div>
            <div className="stat__label">Frameworks Supported</div>
          </div>
          <div className="stat">
            <div className="stat__number">100+</div>
            <div className="stat__label">Community Projects</div>
          </div>
        </div>
      </div>
    </section>
  );
}

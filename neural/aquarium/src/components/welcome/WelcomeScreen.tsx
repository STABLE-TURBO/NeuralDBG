import React, { useState } from 'react';
import QuickStartTemplates from './QuickStartTemplates';
import ExampleGallery from './ExampleGallery';
import DocumentationBrowser from './DocumentationBrowser';
import VideoTutorials from './VideoTutorials';
import './WelcomeScreen.css';

interface WelcomeScreenProps {
  onClose: () => void;
  onLoadTemplate: (template: string) => void;
  onStartTutorial: () => void;
}

type TabType = 'quickstart' | 'examples' | 'docs' | 'videos';

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  onClose,
  onLoadTemplate,
  onStartTutorial,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('quickstart');

  return (
    <div className="welcome-screen-overlay">
      <div className="welcome-screen">
        <div className="welcome-header">
          <div className="welcome-title">
            <h1>Welcome to Neural Aquarium</h1>
            <p>Visual IDE for Neural DSL - Build neural networks with ease</p>
          </div>
          <button className="close-button" onClick={onClose} aria-label="Close welcome screen">
            ×
          </button>
        </div>

        <div className="welcome-actions">
          <button className="action-button primary" onClick={onStartTutorial}>
            🎓 Start Interactive Tutorial
          </button>
          <button className="action-button" onClick={onClose}>
            Skip to IDE
          </button>
        </div>

        <div className="welcome-tabs">
          <button
            className={`tab ${activeTab === 'quickstart' ? 'active' : ''}`}
            onClick={() => setActiveTab('quickstart')}
          >
            Quick Start
          </button>
          <button
            className={`tab ${activeTab === 'examples' ? 'active' : ''}`}
            onClick={() => setActiveTab('examples')}
          >
            Examples
          </button>
          <button
            className={`tab ${activeTab === 'docs' ? 'active' : ''}`}
            onClick={() => setActiveTab('docs')}
          >
            Documentation
          </button>
          <button
            className={`tab ${activeTab === 'videos' ? 'active' : ''}`}
            onClick={() => setActiveTab('videos')}
          >
            Video Tutorials
          </button>
        </div>

        <div className="welcome-content">
          {activeTab === 'quickstart' && (
            <QuickStartTemplates onLoadTemplate={onLoadTemplate} />
          )}
          {activeTab === 'examples' && (
            <ExampleGallery onLoadExample={onLoadTemplate} />
          )}
          {activeTab === 'docs' && <DocumentationBrowser />}
          {activeTab === 'videos' && <VideoTutorials />}
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;

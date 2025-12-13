import React, { useState, useEffect } from 'react';
import './InteractiveTutorial.css';

interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target?: string;
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center';
  highlight?: boolean;
}

interface InteractiveTutorialProps {
  onComplete: () => void;
  onSkip: () => void;
}

const InteractiveTutorial: React.FC<InteractiveTutorialProps> = ({ onComplete, onSkip }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isActive, setIsActive] = useState(true);

  const tutorialSteps: TutorialStep[] = [
    {
      id: 'welcome',
      title: 'Welcome to Neural Aquarium! 🌊',
      content:
        'This interactive tutorial will guide you through the key features of the Neural Aquarium IDE. Let\'s get started!',
      position: 'center',
    },
    {
      id: 'ai-assistant',
      title: 'AI Assistant',
      content:
        'Use the AI Assistant to generate neural networks from natural language descriptions. Just describe what you want to build!',
      target: '.ai-assistant-sidebar',
      position: 'left',
      highlight: true,
    },
    {
      id: 'quick-start',
      title: 'Quick Start Templates',
      content:
        'Choose from pre-built templates for common tasks like image classification, NLP, and time series forecasting.',
      target: '.welcome-tabs',
      position: 'bottom',
      highlight: true,
    },
    {
      id: 'examples',
      title: 'Example Gallery',
      content:
        'Browse and load example models from the repository. Filter by category or search for specific architectures.',
      target: '.welcome-tabs',
      position: 'bottom',
      highlight: true,
    },
    {
      id: 'visual-editor',
      title: 'Visual Network Designer',
      content:
        'Design neural networks visually with a drag-and-drop interface. Add layers, configure parameters, and see your network in real-time.',
      target: '.model-workspace',
      position: 'right',
      highlight: true,
    },
    {
      id: 'dsl-code',
      title: 'Neural DSL Code',
      content:
        'Your network is automatically converted to Neural DSL code. Edit the code directly or use the visual editor - they stay in sync!',
      target: '.model-code',
      position: 'top',
      highlight: true,
    },
    {
      id: 'debugger',
      title: 'Real-time Debugger',
      content:
        'Monitor training in real-time with live plots, activation visualizations, and gradient analysis. Set breakpoints and inspect weights.',
      position: 'center',
    },
    {
      id: 'export',
      title: 'Multi-backend Export',
      content:
        'Export your model to TensorFlow, PyTorch, or ONNX. Deploy to cloud platforms like AWS SageMaker, Google Vertex AI, and more.',
      position: 'center',
    },
    {
      id: 'complete',
      title: 'You\'re All Set! 🎉',
      content:
        'You now know the basics of Neural Aquarium. Start building amazing neural networks! Check the documentation for more details.',
      position: 'center',
    },
  ];

  const currentStepData = tutorialSteps[currentStep];

  useEffect(() => {
    if (currentStepData.target && currentStepData.highlight) {
      const targetElement = document.querySelector(currentStepData.target);
      if (targetElement) {
        targetElement.classList.add('tutorial-highlight');
      }

      return () => {
        if (targetElement) {
          targetElement.classList.remove('tutorial-highlight');
        }
      };
    }
  }, [currentStep, currentStepData]);

  const handleNext = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = () => {
    setIsActive(false);
    onComplete();
  };

  const handleSkipTutorial = () => {
    setIsActive(false);
    onSkip();
  };

  if (!isActive) {
    return null;
  }

  const getTooltipStyle = (): React.CSSProperties => {
    if (currentStepData.position === 'center') {
      return {
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
      };
    }

    return {
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
    };
  };

  return (
    <>
      <div className="tutorial-overlay" />
      <div className="tutorial-tooltip" style={getTooltipStyle()}>
        <div className="tutorial-header">
          <h3>{currentStepData.title}</h3>
          <button className="tutorial-close" onClick={handleSkipTutorial} aria-label="Close tutorial">
            ×
          </button>
        </div>

        <div className="tutorial-body">
          <p>{currentStepData.content}</p>
        </div>

        <div className="tutorial-footer">
          <div className="tutorial-progress">
            <span>
              Step {currentStep + 1} of {tutorialSteps.length}
            </span>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${((currentStep + 1) / tutorialSteps.length) * 100}%` }}
              />
            </div>
          </div>

          <div className="tutorial-actions">
            {currentStep > 0 && (
              <button className="tutorial-btn secondary" onClick={handlePrevious}>
                Previous
              </button>
            )}
            <button className="tutorial-btn secondary" onClick={handleSkipTutorial}>
              Skip Tutorial
            </button>
            <button className="tutorial-btn primary" onClick={handleNext}>
              {currentStep === tutorialSteps.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default InteractiveTutorial;

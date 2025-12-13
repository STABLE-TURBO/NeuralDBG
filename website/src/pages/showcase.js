import React, { useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const showcaseProjects = [
  {
    title: 'Medical Image Classification',
    description: 'CNN-based system for detecting diseases from X-ray images with 95% accuracy',
    author: 'Dr. Sarah Chen',
    organization: 'Stanford Medical AI Lab',
    image: '/img/showcase/medical.jpg',
    tags: ['Healthcare', 'CNN', 'Production'],
    githubUrl: 'https://github.com/example/medical-ai',
    demoUrl: 'https://demo.example.com',
    caseStudyUrl: '/blog/case-study-medical-imaging',
  },
  {
    title: 'Real-time Sentiment Analysis',
    description: 'LSTM model analyzing customer feedback in real-time for Fortune 500 company',
    author: 'Alex Rodriguez',
    organization: 'TechCorp Analytics',
    image: '/img/showcase/sentiment.jpg',
    tags: ['NLP', 'LSTM', 'Enterprise'],
    githubUrl: 'https://github.com/example/sentiment-analysis',
    caseStudyUrl: '/blog/case-study-sentiment-analysis',
  },
  {
    title: 'Autonomous Vehicle Vision',
    description: 'Object detection and tracking system for self-driving cars',
    author: 'Michael Zhang',
    organization: 'AutoDrive Labs',
    image: '/img/showcase/autonomous.jpg',
    tags: ['Computer Vision', 'Real-time', 'Production'],
    githubUrl: 'https://github.com/example/autodrive-vision',
    paperUrl: 'https://arxiv.org/example',
  },
  {
    title: 'Financial Fraud Detection',
    description: 'Neural network detecting fraudulent transactions with 99.2% precision',
    author: 'Emily Johnson',
    organization: 'SecureBank AI',
    image: '/img/showcase/fraud.jpg',
    tags: ['Finance', 'Anomaly Detection', 'Production'],
    githubUrl: 'https://github.com/example/fraud-detection',
    caseStudyUrl: '/blog/case-study-fraud-detection',
  },
  {
    title: 'Speech Recognition System',
    description: 'Multi-language speech-to-text with transformer architecture',
    author: 'David Park',
    organization: 'VoiceAI Research',
    image: '/img/showcase/speech.jpg',
    tags: ['Audio', 'Transformer', 'Research'],
    githubUrl: 'https://github.com/example/speech-recognition',
    paperUrl: 'https://arxiv.org/example',
  },
  {
    title: 'E-commerce Recommendation Engine',
    description: 'Personalized product recommendations serving 10M+ users daily',
    author: 'Lisa Wang',
    organization: 'ShopMart',
    image: '/img/showcase/ecommerce.jpg',
    tags: ['Recommender Systems', 'Production', 'Scale'],
    githubUrl: 'https://github.com/example/recommendation-engine',
    caseStudyUrl: '/blog/case-study-ecommerce-recommendations',
  },
  {
    title: 'Climate Change Prediction',
    description: 'Time-series forecasting for climate patterns using RNN architecture',
    author: 'Prof. James Miller',
    organization: 'Climate Research Institute',
    image: '/img/showcase/climate.jpg',
    tags: ['Time Series', 'Research', 'RNN'],
    githubUrl: 'https://github.com/example/climate-prediction',
    paperUrl: 'https://arxiv.org/example',
  },
  {
    title: 'Educational Chatbot',
    description: 'AI tutor helping students learn programming concepts',
    author: 'Maria Garcia',
    organization: 'EduTech Solutions',
    image: '/img/showcase/chatbot.jpg',
    tags: ['NLP', 'Education', 'Chatbot'],
    githubUrl: 'https://github.com/example/edu-chatbot',
    demoUrl: 'https://demo.example.com',
  },
  {
    title: 'Industrial Quality Control',
    description: 'Defect detection in manufacturing with 99.5% accuracy',
    author: 'Robert Lee',
    organization: 'ManufacturePro',
    image: '/img/showcase/quality.jpg',
    tags: ['Computer Vision', 'Manufacturing', 'Production'],
    caseStudyUrl: '/blog/case-study-quality-control',
  },
];

const allTags = [...new Set(showcaseProjects.flatMap(p => p.tags))].sort();

function ShowcaseCard({ project }) {
  return (
    <div className="showcase-item">
      <div 
        className="showcase-item__image" 
        style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: '3rem'
        }}>
        {project.title.charAt(0)}
      </div>
      <div className="showcase-item__content">
        <h3 className="showcase-item__title">{project.title}</h3>
        <p className="showcase-item__description">{project.description}</p>
        <p style={{fontSize: '0.9rem', marginBottom: '1rem'}}>
          <strong>{project.author}</strong> · {project.organization}
        </p>
        <div className="showcase-item__tags">
          {project.tags.map((tag, idx) => (
            <span key={idx} className="showcase-item__tag">{tag}</span>
          ))}
        </div>
        <div style={{marginTop: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap'}}>
          {project.githubUrl && (
            <a href={project.githubUrl} className="button button--sm button--outline button--primary" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
          )}
          {project.demoUrl && (
            <a href={project.demoUrl} className="button button--sm button--outline button--primary" target="_blank" rel="noopener noreferrer">
              Demo
            </a>
          )}
          {project.caseStudyUrl && (
            <Link to={project.caseStudyUrl} className="button button--sm button--primary">
              Case Study
            </Link>
          )}
          {project.paperUrl && (
            <a href={project.paperUrl} className="button button--sm button--outline button--primary" target="_blank" rel="noopener noreferrer">
              Paper
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Showcase() {
  const [selectedTag, setSelectedTag] = useState('All');
  
  const filteredProjects = selectedTag === 'All' 
    ? showcaseProjects 
    : showcaseProjects.filter(p => p.tags.includes(selectedTag));

  return (
    <Layout
      title="Showcase"
      description="Discover amazing projects built with Neural DSL">
      <div className="container margin-vert--lg">
        <div style={{textAlign: 'center', marginBottom: '3rem'}}>
          <h1>Community Showcase</h1>
          <p style={{fontSize: '1.25rem', color: 'var(--ifm-color-emphasis-600)'}}>
            Discover amazing projects built with Neural DSL by our community
          </p>
        </div>

        <div style={{marginBottom: '2rem', textAlign: 'center'}}>
          <div style={{display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center'}}>
            <button
              className={`button button--sm ${selectedTag === 'All' ? 'button--primary' : 'button--outline button--primary'}`}
              onClick={() => setSelectedTag('All')}>
              All ({showcaseProjects.length})
            </button>
            {allTags.map(tag => (
              <button
                key={tag}
                className={`button button--sm ${selectedTag === tag ? 'button--primary' : 'button--outline button--primary'}`}
                onClick={() => setSelectedTag(tag)}>
                {tag} ({showcaseProjects.filter(p => p.tags.includes(tag)).length})
              </button>
            ))}
          </div>
        </div>

        <div className="showcase-grid">
          {filteredProjects.map((project, idx) => (
            <ShowcaseCard key={idx} project={project} />
          ))}
        </div>

        <div style={{textAlign: 'center', marginTop: '4rem', padding: '3rem', background: 'var(--ifm-color-emphasis-100)', borderRadius: '8px'}}>
          <h2>Submit Your Project</h2>
          <p style={{fontSize: '1.1rem', marginBottom: '2rem'}}>
            Built something awesome with Neural DSL? Share it with the community!
          </p>
          <a
            className="button button--primary button--lg"
            href="https://github.com/Lemniscate-world/Neural/issues/new?template=showcase.md"
            target="_blank"
            rel="noopener noreferrer">
            Submit Your Project
          </a>
        </div>
      </div>
    </Layout>
  );
}

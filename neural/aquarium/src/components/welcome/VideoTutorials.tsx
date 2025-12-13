import React, { useState } from 'react';
import './VideoTutorials.css';

interface Video {
  id: string;
  title: string;
  description: string;
  duration: string;
  category: string;
  embedUrl: string;
  thumbnail: string;
  level: 'beginner' | 'intermediate' | 'advanced';
}

const VideoTutorials: React.FC = () => {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const videos: Video[] = [
    {
      id: 'intro',
      title: 'Introduction to Neural Aquarium',
      description: 'Get started with the Neural Aquarium IDE and learn the basics of Neural DSL',
      duration: '10:30',
      category: 'Getting Started',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EIntroduction%3C/text%3E%3C/svg%3E',
      level: 'beginner',
    },
    {
      id: 'dsl-basics',
      title: 'Neural DSL Syntax Basics',
      description: 'Learn the fundamental syntax and structure of Neural DSL language',
      duration: '15:45',
      category: 'Language',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EDSL Basics%3C/text%3E%3C/svg%3E',
      level: 'beginner',
    },
    {
      id: 'first-model',
      title: 'Building Your First Model',
      description: 'Step-by-step guide to creating your first neural network with Neural DSL',
      duration: '20:15',
      category: 'Tutorial',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EFirst Model%3C/text%3E%3C/svg%3E',
      level: 'beginner',
    },
    {
      id: 'ai-assistant',
      title: 'Using the AI Assistant',
      description: 'Leverage AI to generate neural network architectures from natural language',
      duration: '12:00',
      category: 'Features',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EAI Assistant%3C/text%3E%3C/svg%3E',
      level: 'beginner',
    },
    {
      id: 'debugger',
      title: 'Debugging Neural Networks',
      description: 'Master the real-time debugger for training visualization and monitoring',
      duration: '18:30',
      category: 'Features',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EDebugger%3C/text%3E%3C/svg%3E',
      level: 'intermediate',
    },
    {
      id: 'cnn-guide',
      title: 'Convolutional Neural Networks',
      description: 'Deep dive into building CNNs for computer vision tasks',
      duration: '25:00',
      category: 'Tutorial',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3ECNNs%3C/text%3E%3C/svg%3E',
      level: 'intermediate',
    },
    {
      id: 'rnn-guide',
      title: 'Recurrent Neural Networks',
      description: 'Learn to build RNNs and LSTMs for sequence modeling',
      duration: '22:45',
      category: 'Tutorial',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3ERNNs%3C/text%3E%3C/svg%3E',
      level: 'intermediate',
    },
    {
      id: 'deployment',
      title: 'Deploying to Production',
      description: 'Export and deploy your models to various ML platforms',
      duration: '16:20',
      category: 'Advanced',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EDeployment%3C/text%3E%3C/svg%3E',
      level: 'advanced',
    },
    {
      id: 'hpo',
      title: 'Hyperparameter Optimization',
      description: 'Automate hyperparameter tuning with built-in HPO tools',
      duration: '19:10',
      category: 'Advanced',
      embedUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      thumbnail: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="320" height="180"%3E%3Crect fill="%232a2a2a"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%2361dafb" font-size="20" font-family="sans-serif"%3EHPO%3C/text%3E%3C/svg%3E',
      level: 'advanced',
    },
  ];

  const categories = ['all', ...new Set(videos.map((v) => v.category))];

  const filteredVideos = videos.filter(
    (video) => selectedCategory === 'all' || video.category === selectedCategory
  );

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'beginner':
        return '#4caf50';
      case 'intermediate':
        return '#ff9800';
      case 'advanced':
        return '#f44336';
      default:
        return '#999';
    }
  };

  return (
    <div className="video-tutorials">
      <div className="videos-header">
        <h2>Video Tutorials</h2>
        <p>Watch step-by-step guides to master Neural Aquarium</p>
      </div>

      <div className="video-categories">
        {categories.map((category) => (
          <button
            key={category}
            className={`video-category ${selectedCategory === category ? 'active' : ''}`}
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </button>
        ))}
      </div>

      {selectedVideo && (
        <div className="video-player-container">
          <div className="video-player">
            <button
              className="close-player"
              onClick={() => setSelectedVideo(null)}
              aria-label="Close video"
            >
              ×
            </button>
            <iframe
              src={videos.find((v) => v.id === selectedVideo)?.embedUrl}
              title={videos.find((v) => v.id === selectedVideo)?.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </div>
        </div>
      )}

      <div className="videos-grid">
        {filteredVideos.map((video) => (
          <div key={video.id} className="video-card">
            <div
              className="video-thumbnail"
              onClick={() => setSelectedVideo(video.id)}
              style={{ backgroundImage: `url(${video.thumbnail})` }}
            >
              <div className="play-overlay">
                <div className="play-button">▶</div>
              </div>
              <div className="video-duration">{video.duration}</div>
            </div>
            <div className="video-info">
              <div className="video-header-section">
                <h3>{video.title}</h3>
                <span
                  className="level-badge"
                  style={{ backgroundColor: getLevelColor(video.level) }}
                >
                  {video.level}
                </span>
              </div>
              <p className="video-category">{video.category}</p>
              <p className="video-description">{video.description}</p>
              <button
                className="watch-video-btn"
                onClick={() => setSelectedVideo(video.id)}
              >
                Watch Now
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default VideoTutorials;

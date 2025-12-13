# Welcome Screen Components

This directory contains the welcome screen and tutorial system for Neural Aquarium IDE.

## Components

### WelcomeScreen.tsx
Main welcome screen component that serves as the entry point for new users. Features:
- Tabbed interface with Quick Start, Examples, Documentation, and Video Tutorials
- Interactive tutorial launcher
- Skip to IDE option
- Responsive design with smooth animations

### QuickStartTemplates.tsx
Pre-built neural network templates for common use cases:
- **Image Classification** (CNN for MNIST/CIFAR-10) - Beginner
- **Text Classification** (LSTM for NLP) - Beginner
- **Time Series Forecasting** (LSTM) - Intermediate
- **Autoencoder** (Deep autoencoder) - Intermediate
- **Sequence-to-Sequence** (Encoder-decoder) - Advanced
- **GAN** (Generative Adversarial Network) - Advanced

Each template includes:
- Pre-written Neural DSL code
- Difficulty level badge
- Category classification
- One-click loading

### ExampleGallery.tsx
Browse and load example models from the examples/ directory:
- Dynamic loading from backend API
- Category filtering (Computer Vision, NLP, Generative)
- Search functionality
- Tag-based organization
- Fallback to built-in examples if API unavailable

### DocumentationBrowser.tsx
Integrated documentation viewer:
- Markdown rendering with react-markdown
- Categorized documentation sections
- Search functionality
- Sidebar navigation
- Placeholder content with fallback to API

Categories:
- **Getting Started**: Quick Start Guide
- **Language**: DSL Syntax Reference, Layer Types
- **Tools**: Debugger Features
- **Advanced**: Deployment Guide, Platform Integrations

### VideoTutorials.tsx
Video tutorial library:
- Video embeds with YouTube support
- Category filtering (Getting Started, Language, Tutorial, Features, Advanced)
- Difficulty levels (Beginner, Intermediate, Advanced)
- Modal video player
- Duration display
- Thumbnail previews

### InteractiveTutorial.tsx
Step-by-step guided tour using Shepherd.js-inspired overlay system:
- 9-step tutorial covering all major features
- Element highlighting with animations
- Progress tracking
- Previous/Next/Skip navigation
- Centered and contextual positioning
- Auto-cleanup of highlights

Tutorial Steps:
1. Welcome message
2. AI Assistant introduction
3. Quick Start Templates
4. Example Gallery
5. Visual Network Designer
6. DSL Code editor
7. Real-time Debugger
8. Multi-backend Export
9. Completion message

## Usage

### Integration with App.tsx

```tsx
import { WelcomeScreen, InteractiveTutorial } from './components/welcome';

function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [showTutorial, setShowTutorial] = useState(false);

  const handleLoadTemplate = (dslCode: string) => {
    // Load template into editor
    setShowWelcome(false);
  };

  const handleStartTutorial = () => {
    setShowWelcome(false);
    setShowTutorial(true);
  };

  return (
    <>
      {showWelcome && (
        <WelcomeScreen
          onClose={() => setShowWelcome(false)}
          onLoadTemplate={handleLoadTemplate}
          onStartTutorial={handleStartTutorial}
        />
      )}
      
      {showTutorial && (
        <InteractiveTutorial
          onComplete={() => setShowTutorial(false)}
          onSkip={() => setShowTutorial(false)}
        />
      )}
      
      {/* Rest of app */}
    </>
  );
}
```

## API Endpoints

The components integrate with the following backend endpoints:

### Examples API
- `GET /api/examples/list` - List all available examples
- `GET /api/examples/load?path={path}` - Load example code
- `GET /api/examples/search?q={query}&category={category}` - Search examples

### Documentation API
- `GET /api/docs/{doc_path}` - Load documentation file
- `GET /api/docs/list` - List all documentation files
- `GET /api/docs/search?q={query}` - Search documentation

## Styling

All components use consistent styling:
- Dark theme (#1e1e1e, #2a2a2a backgrounds)
- Neural Aquarium blue (#61dafb) for accents
- Smooth animations and transitions
- Responsive grid layouts
- Custom scrollbars
- Hover effects and visual feedback

## Dependencies

- React 18.2+
- TypeScript 4.9+
- axios (API requests)
- react-markdown (documentation rendering)

## File Structure

```
welcome/
├── WelcomeScreen.tsx          # Main welcome component
├── WelcomeScreen.css          # Main welcome styles
├── QuickStartTemplates.tsx    # Template gallery
├── QuickStartTemplates.css    # Template styles
├── ExampleGallery.tsx         # Example browser
├── ExampleGallery.css         # Example styles
├── DocumentationBrowser.tsx   # Docs viewer
├── DocumentationBrowser.css   # Docs styles
├── VideoTutorials.tsx         # Video library
├── VideoTutorials.css         # Video styles
├── InteractiveTutorial.tsx    # Guided tour
├── InteractiveTutorial.css    # Tutorial styles
├── index.tsx                  # Export barrel
└── README.md                  # This file
```

## Customization

### Adding New Templates
Edit `QuickStartTemplates.tsx` and add to the `templates` array:

```tsx
{
  id: 'my-template',
  title: 'My Template',
  description: 'Description here',
  category: 'Category',
  icon: '🎯',
  difficulty: 'beginner' | 'intermediate' | 'advanced',
  dslCode: `network MyNetwork { ... }`
}
```

### Adding Tutorial Steps
Edit `InteractiveTutorial.tsx` and add to the `tutorialSteps` array:

```tsx
{
  id: 'my-step',
  title: 'My Step Title',
  content: 'Step description',
  target: '.css-selector',  // Optional
  position: 'top',          // Optional
  highlight: true           // Optional
}
```

### Adding Video Tutorials
Edit `VideoTutorials.tsx` and add to the `videos` array:

```tsx
{
  id: 'my-video',
  title: 'My Video',
  description: 'Video description',
  duration: '10:30',
  category: 'Tutorial',
  embedUrl: 'https://youtube.com/embed/...',
  thumbnail: 'path/to/thumbnail.jpg',
  level: 'beginner'
}
```

## Accessibility

All components include:
- ARIA labels for buttons and interactive elements
- Keyboard navigation support
- High contrast colors
- Clear focus indicators
- Screen reader compatible markup

## Future Enhancements

- [ ] Local storage for "Don't show again" preference
- [ ] Progress tracking for completed tutorials
- [ ] User favorites for templates and examples
- [ ] Offline documentation support
- [ ] Multiple language support
- [ ] Video playback progress tracking
- [ ] Custom template creation and sharing
- [ ] Interactive code playground in tutorials

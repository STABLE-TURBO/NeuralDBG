import React, { useState } from 'react';
import NetworkDesigner from './components/designer/NetworkDesigner';
import './App.css';

const App: React.FC = () => {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Neural Aquarium - Visual Network Designer</h1>
      </header>
      <NetworkDesigner />
    </div>
  );
};

export default App;

import React, { useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import './CodeEditor.css';

interface CodeEditorProps {
  code: string;
  onChange: (code: string) => void;
}

const CodeEditor: React.FC<CodeEditorProps> = ({ code, onChange }) => {
  const handleEditorChange = (value: string | undefined) => {
    if (value !== undefined) {
      onChange(value);
    }
  };

  return (
    <div className="code-editor">
      <div className="editor-header">
        <h3>Neural DSL Code</h3>
        <div className="editor-actions">
          <button 
            className="action-btn"
            onClick={() => navigator.clipboard.writeText(code)}
            title="Copy to clipboard"
          >
            📋 Copy
          </button>
        </div>
      </div>
      <Editor
        height="100%"
        defaultLanguage="python"
        theme="vs-dark"
        value={code}
        onChange={handleEditorChange}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 4,
          wordWrap: 'on',
        }}
      />
    </div>
  );
};

export default CodeEditor;
